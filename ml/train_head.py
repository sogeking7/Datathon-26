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
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from common import load_labels, window_targets

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


def build_examples(cache_dir: Path, videos: dict) -> list[dict]:
    ex = []
    for p in sorted(cache_dir.glob("*.pt")):
        d = torch.load(p)
        v = videos.get(d["file"])
        if v is None:  # дубль (common.DUPLICATES)
            continue
        for key, h in d["hidden"].items():
            task, code = key.split(":", 1)
            q = BY_CODE[code]
            y = window_targets(v, d["start"], d["end"], code, q.state)
            if y is None:
                continue
            if task == "choice":
                target = q.yes_index if y else (1 if q.yes_index == 0 else 0)
                ex.append(dict(file=d["file"], aug=d["aug"], start=d["start"], end=d["end"], code=code,
                               task=task, pairs=h["pairs"], query=h["query"], target=target))
            elif y:  # grounding учим только там, где событие есть
                times = torch.tensor(d["ground_times"][code], dtype=torch.float32)  # от начала видео
                t = min((e["timecode_sec"] for e in v["events"] if e["type_code"] == code),
                        key=lambda x: abs(x - (d["start"] + d["end"]) / 2))
                lo, hi = (t, t + 2 * GROUND_HALF_SEC) if q.state else (t - GROUND_HALF_SEC, t + GROUND_HALF_SEC)
                target = ((times >= lo) & (times <= hi)).float()
                if target.sum() == 0:
                    target[(times - t).abs().argmin()] = 1
                ex.append(dict(file=d["file"], aug=d["aug"], start=d["start"], end=d["end"], code=code,
                               task=task, pairs=h["pairs"], query=h["query"], target=target))
    return ex


def train(examples, init, epochs, lr, anchor, ground_w, seed=0, device="cuda"):
    torch.manual_seed(seed)
    random.seed(seed)
    head = Head().to(device)
    head.load_state_dict(init)
    ref = {k: v.detach().clone().to(device) for k, v in head.state_dict().items()}
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    # Балансируем классы гейтов: «да» в данных намного реже, а пропуск нарушения хуже ложного срабатывания.
    pos = sum(1 for e in examples if e["task"] == "choice" and e["target"] == BY_CODE[e["code"]].yes_index)
    neg = sum(1 for e in examples if e["task"] == "choice") - pos
    w_pos = (neg / max(pos, 1)) ** 0.5
    for _ in range(epochs):
        random.shuffle(examples)
        for i in range(0, len(examples), 32):
            loss = 0.0
            for e in examples[i:i + 32]:
                logits = head(e["pairs"].to(device), e["query"].to(device))
                if e["task"] == "choice":
                    w = w_pos if e["target"] == BY_CODE[e["code"]].yes_index else 1.0
                    loss = loss + w * F.cross_entropy(logits[None], torch.tensor([e["target"]], device=device))
                else:
                    loss = loss + ground_w * F.binary_cross_entropy_with_logits(logits, e["target"].to(device))
            # Держим голову рядом с предобученной: данных мало, иначе забудет общие знания.
            reg = sum(((p - ref[n]) ** 2).sum() for n, p in head.named_parameters())
            loss = loss / len(examples[i:i + 32]) + anchor * reg
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head.eval()


@torch.no_grad()
def rescore(head, rows_for_video, cache_dir, device="cuda"):
    """Пересчитывает gate/ground у записей scores.jsonl (aug=none) новой головой."""
    out = []
    for r in rows_for_video:
        wi = r["_wi"]
        d = torch.load(cache_dir / f"{Path(r['file']).stem}__none__{wi}.pt")
        r = {k: v for k, v in r.items() if k != "_wi"}
        gate, ground = {}, {}
        for key, h in d["hidden"].items():
            task, code = key.split(":", 1)
            logits = head(h["pairs"].to(device), h["query"].to(device))
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
    a = p.parse_args()

    cache, out = Path(a.cache), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    videos = {v["file"]: v for v in load_labels()}
    init = pretrained_head(a.model)
    examples = build_examples(cache, videos)
    print(f"examples: {len(examples)} from {len({e['file'] for e in examples})} videos", flush=True)

    rows = defaultdict(list)
    counters = defaultdict(int)
    for line in Path(a.scores).read_text().splitlines():
        r = json.loads(line)
        if r["aug"] == "none" and r["file"] in videos:
            r["_wi"] = counters[r["file"]]
            counters[r["file"]] += 1
            rows[r["file"]].append(r)

    if not a.no_cv:
        with (out / "scores_cv.jsonl").open("w") as f:
            for i, file in enumerate(sorted(rows)):
                head = train([e for e in examples if e["file"] != file], init,
                             a.epochs, a.lr, a.anchor, a.ground_w, seed=i)
                for r in rescore(head, rows[file], cache):
                    f.write(json.dumps(r) + "\n")
                print(f"cv {i + 1}/{len(rows)} {file}", flush=True)
    head = train(list(examples), init, a.epochs, a.lr, a.anchor, a.ground_w)
    torch.save({k: v.cpu() for k, v in head.state_dict().items()}, out / "head.pt")
    (out / "train_args.json").write_text(json.dumps(vars(a), indent=1))
    print("saved", out / "head.pt")


if __name__ == "__main__":
    main()
