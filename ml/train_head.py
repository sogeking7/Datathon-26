"""Дообучение decision head VideoJev на нашем датасете (аналог официального этапа 1 «head warmup»).

Почему только голова: официальный этап 2 (весь 4B-бэкбон, DeepSpeed ZeRO-3, 8×H200) на одной T4
невозможен, а на 42 роликах он бы переобучился. Голова — это два линейных слоя 2560→512 (~2.6 млн
параметров) поверх скрытых состояний замороженной модели. Скрытые состояния считаются один раз
(score_dataset.py --hidden), дальше обучение занимает секунды и идёт с leave-one-video-out.

python ml/train_head.py --cache runs/cache --model /models/videojev-4b-fp16 \
    --scores runs/zeroshot/scores.jsonl --out runs/head
Результат:
  runs/head/scores_cv.jsonl — оценки окон от головы, не видевшей это видео (вход для calibrate.py);
  runs/head/head.pt         — голова, обученная на всех видео (VIDEOJEV_HEAD в сервисе).

Параметры по умолчанию воспроизводят голову первого выпуска. Экспериментальные ключи
(см. reports/experiment_plan.md):
  --alt-policy ignore|positive  окна альтернативных типов (labels.json alt_type_codes) не учатся как «0»;
  --split setup                 кросс-валидация по сессиям съёмки (analysis/setups.json), а не по роликам;
  --seeds N --ensemble soup     N голов с разными seed, усреднение весов (одна голова на выходе);
  --feature-noise σ / --feature-dropout p   аугментация в пространстве признаков (FroFA);
  --pos-power 0.5               степень веса положительного класса (neg/pos)^power;
  --exclude-type CODE           leave-one-type-out: ролики с этим типом (основным или альтернативным)
                                не участвуют в обучении, чтобы измерить поведение на неизвестном типе.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from common import load_labels, load_setups, video_codes, window_targets

from app.vj.queries import BY_CODE

GROUND_HALF_SEC = 1.5  # размеченный таймкод → отрезок ±1.5 с для grounding


class Head(nn.Module):
    """Копия CosineDecisionHead из modeling_videojev.py."""

    def __init__(self, hidden=2560, dim=512, scale=10.0):
        super().__init__()
        self.q = nn.Linear(hidden, dim)
        self.k = nn.Linear(hidden, dim)
        self.bias = nn.Parameter(torch.zeros(()))
        self.scale = scale

    def forward(self, pairs, query):  # pairs (K, H), query (H) → logits (K)
        k = F.normalize(self.k(pairs.float()), dim=-1, eps=1e-6)
        q = F.normalize(self.q(query.float()), dim=-1, eps=1e-6)
        return k @ q * self.scale + self.bias


def pretrained_head(model_dir: str) -> dict:
    from safetensors import safe_open

    index = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    state = {}
    for name, shard in index.items():
        if name.startswith("head."):
            with safe_open(str(Path(model_dir) / shard), "pt") as f:
                state[name[len("head."):]] = f.get_tensor(name).float()
    return state


def build_examples(cache_dir: Path, videos: dict, alt_policy: str = "negative", exclude_type: str | None = None) -> list[dict]:
    ex = []
    for p in sorted(cache_dir.glob("*.pt")):
        d = torch.load(p)
        v = videos.get(d["file"])
        if v is None:  # дубль (common.DUPLICATES)
            continue
        if exclude_type and exclude_type in video_codes(v):
            continue
        for key, h in d["hidden"].items():
            task, code = key.split(":", 1)
            q = BY_CODE[code]
            y = window_targets(v, d["start"], d["end"], code, q.state, alt_policy=alt_policy)
            if y is None:
                continue
            if task == "choice":
                target = q.yes_index if y else (1 if q.yes_index == 0 else 0)
                ex.append(dict(file=d["file"], aug=d["aug"], start=d["start"], end=d["end"], code=code,
                               task=task, pairs=h["pairs"], query=h["query"], target=target))
            elif y:  # grounding учим только там, где событие есть
                times = torch.tensor(d["ground_times"][code], dtype=torch.float32)  # от начала видео
                cands = [e["timecode_sec"] for e in v["events"]
                         if e["type_code"] == code or (alt_policy == "positive" and code in e.get("alt_type_codes", []))]
                t = min(cands, key=lambda x: abs(x - (d["start"] + d["end"]) / 2))
                lo, hi = (t, t + 2 * GROUND_HALF_SEC) if q.state else (t - GROUND_HALF_SEC, t + GROUND_HALF_SEC)
                target = ((times >= lo) & (times <= hi)).float()
                if target.sum() == 0:
                    target[(times - t).abs().argmin()] = 1
                ex.append(dict(file=d["file"], aug=d["aug"], start=d["start"], end=d["end"], code=code,
                               task=task, pairs=h["pairs"], query=h["query"], target=target))
    return ex


def train(examples, init, epochs, lr, anchor, ground_w, seed=0, device="cuda", pos_power=0.5,
          feature_noise=0.0, feature_dropout=0.0):
    torch.manual_seed(seed)
    random.seed(seed)
    head = Head().to(device)
    head.load_state_dict(init)
    ref = {k: v.detach().clone().to(device) for k, v in head.state_dict().items()}
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    # Балансируем классы гейтов: «да» в данных намного реже, а пропуск нарушения хуже ложного срабатывания.
    pos = sum(1 for e in examples if e["task"] == "choice" and e["target"] == BY_CODE[e["code"]].yes_index)
    neg = sum(1 for e in examples if e["task"] == "choice") - pos
    w_pos = (neg / max(pos, 1)) ** pos_power

    def features(x):
        x = x.float()
        if feature_noise:
            x = x + feature_noise * x.std() * torch.randn_like(x)
        if feature_dropout:
            x = F.dropout(x, feature_dropout, training=True)
        return x

    # Один батч — один проход: пары дополняются до общей длины K, потери считаются по маске.
    # Математически то же, что поэкземплярный цикл первого выпуска (сумма потерь / размер батча + якорь),
    # но в ~30 раз быстрее: 40 фолдов LOVO укладываются в минуты вместо часа.
    examples = list(examples)
    dev_pairs = [e["pairs"].to(device) for e in examples]
    dev_query = [e["query"].to(device) for e in examples]
    for _ in range(epochs):
        order = list(range(len(examples)))
        random.shuffle(order)
        for i in range(0, len(order), 32):
            idx = order[i:i + 32]
            kmax = max(dev_pairs[j].shape[0] for j in idx)
            pairs = torch.zeros(len(idx), kmax, dev_pairs[idx[0]].shape[1], device=device)
            mask = torch.zeros(len(idx), kmax, dtype=torch.bool, device=device)
            for b, j in enumerate(idx):
                k = dev_pairs[j].shape[0]
                pairs[b, :k] = features(dev_pairs[j])
                mask[b, :k] = True
            query = features(torch.stack([dev_query[j] for j in idx]))
            keys = F.normalize(head.k(pairs), dim=-1, eps=1e-6)
            q = F.normalize(head.q(query), dim=-1, eps=1e-6)
            logits = torch.einsum("bkd,bd->bk", keys, q) * head.scale + head.bias
            loss = torch.zeros((), device=device)
            for b, j in enumerate(idx):
                e = examples[j]
                k = dev_pairs[j].shape[0]
                if e["task"] == "choice":
                    w = w_pos if e["target"] == BY_CODE[e["code"]].yes_index else 1.0
                    loss = loss + w * F.cross_entropy(logits[b, :k][None], torch.tensor([e["target"]], device=device))
                else:
                    loss = loss + ground_w * F.binary_cross_entropy_with_logits(logits[b, :k], e["target"].to(device))
            # Держим голову рядом с предобученной: данных мало, иначе забудет общие знания.
            reg = sum(((p - ref[n]) ** 2).sum() for n, p in head.named_parameters())
            loss = loss / len(idx) + anchor * reg
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head.eval()


def soup(heads: list[Head]) -> Head:
    """Усреднение весов голов, обученных из одной инициализации (model soup); одна голова на выходе."""
    out = Head().to(next(heads[0].parameters()).device)
    state = {k: sum(h.state_dict()[k].float() for h in heads) / len(heads) for k in heads[0].state_dict()}
    out.load_state_dict(state)
    return out.eval()


def train_ensemble(examples, init, a, seed_base=0, device="cuda") -> list[Head]:
    heads = [train(examples, init, a.epochs, a.lr, a.anchor, a.ground_w, seed=seed_base * 1000 + s, device=device,
                   pos_power=a.pos_power, feature_noise=a.feature_noise, feature_dropout=a.feature_dropout)
             for s in range(a.seeds)]
    return [soup(heads)] if a.ensemble == "soup" and len(heads) > 1 else heads


@torch.no_grad()
def rescore(heads, rows_for_video, cache_dir, device="cuda"):
    """Пересчитывает gate/ground у записей scores.jsonl (aug=none); несколько голов → среднее логитов."""
    heads = heads if isinstance(heads, list) else [heads]
    out = []
    for r in rows_for_video:
        wi = r["_wi"]
        d = torch.load(cache_dir / f"{Path(r['file']).stem}__none__{wi}.pt")
        r = {k: v for k, v in r.items() if k != "_wi"}
        gate, ground = {}, {}
        for key, h in d["hidden"].items():
            task, code = key.split(":", 1)
            logits = sum(head(h["pairs"].to(device), h["query"].to(device)) for head in heads) / len(heads)
            if task == "choice":
                gate[code] = logits.softmax(-1).cpu().tolist()
            else:
                ground[code] = dict(times=r["ground"][code]["times"], probs=logits.sigmoid().cpu().tolist())
        out.append({**r, "gate": gate, "ground": ground})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--scores", required=True, help="scores.jsonl того же прогона (aug=none)")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--anchor", type=float, default=1e-3)
    p.add_argument("--ground-w", type=float, default=0.5)
    p.add_argument("--no-cv", action="store_true")
    # экспериментальные ключи; значения по умолчанию = голова первого выпуска
    p.add_argument("--alt-policy", default="negative", choices=["negative", "ignore", "positive"])
    p.add_argument("--split", default="video", choices=["video", "setup"])
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--ensemble", default="soup", choices=["soup", "logits"])
    p.add_argument("--pos-power", type=float, default=0.5)
    p.add_argument("--feature-noise", type=float, default=0.0)
    p.add_argument("--feature-dropout", type=float, default=0.0)
    p.add_argument("--exclude-type", default=None)
    p.add_argument("--augs", default=None, help="через запятую: какие аугментации из кэша использовать (по умолчанию все)")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    cache, out = Path(a.cache), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    videos = {v["file"]: v for v in load_labels()}
    init = pretrained_head(a.model)
    examples = build_examples(cache, videos, alt_policy=a.alt_policy, exclude_type=a.exclude_type)
    if a.augs:
        keep = set(a.augs.split(","))
        examples = [e for e in examples if e["aug"] in keep]
    print(f"examples: {len(examples)} from {len({e['file'] for e in examples})} videos", flush=True)

    rows = defaultdict(list)
    counters = defaultdict(int)
    for line in Path(a.scores).read_text().splitlines():
        r = json.loads(line)
        if r["aug"] == "none" and r["file"] in videos:
            r["_wi"] = counters[r["file"]]
            counters[r["file"]] += 1
            rows[r["file"]].append(r)

    group_of = (lambda f: f) if a.split == "video" else (lambda f, s=load_setups(): s.get(f, f))
    if not a.no_cv:
        folds = sorted({group_of(f) for f in rows})
        # scores_cv.jsonl: held-out videos scored by the head that never saw them (test scores);
        # scores_cv_train.jsonl: the fold's TRAINING videos scored by the same head, with a "fold" field, so that
        # thresholds can be fitted per fold on the very head that will be applied (ml/experiment.py --train-scores).
        # Without it thresholds are fitted on a mixture of other folds' heads and may not transfer (audit F15).
        with (out / "scores_cv.jsonl").open("w") as f, (out / "scores_cv_train.jsonl").open("w") as ft:
            for i, fold in enumerate(folds):
                heads = train_ensemble([e for e in examples if group_of(e["file"]) != fold], init, a, seed_base=i,
                                       device=a.device)
                for file in sorted(rows):
                    scored = rescore(heads, rows[file], cache, device=a.device)
                    if group_of(file) == fold:
                        for r in scored:
                            f.write(json.dumps(r) + "\n")
                    else:
                        for r in scored:
                            ft.write(json.dumps({**r, "fold": fold}) + "\n")
                print(f"cv {i + 1}/{len(folds)} {fold}", flush=True)
    heads = train_ensemble(list(examples), init, a, device=a.device)
    if len(heads) > 1:
        print("warning: --ensemble logits даёт несколько голов; в head.pt сохраняется их soup")
        heads = [soup(heads)]
    torch.save({k: v.cpu() for k, v in heads[0].state_dict().items()}, out / "head.pt")
    (out / "train_args.json").write_text(json.dumps(vars(a), indent=1))
    print("saved", out / "head.pt")


if __name__ == "__main__":
    main()
