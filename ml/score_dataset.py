"""Прогон VideoJev по датасету: вероятности гейтов и кривые grounding по всем типам и окнам.

python ml/score_dataset.py --videos /data/dataset --model /models/videojev-4b-fp16 \
    --out runs/zeroshot/scores.jsonl [--augs none,hflip] [--head head.pt] [--hidden runs/cache]

Одна запись JSONL = одно окно одного видео (с одной аугментацией). С --hidden дополнительно
сохраняет скрытые состояния, которые подаются в decision head, — на них учится train_head.py.
"""

import argparse
import json
import time
from pathlib import Path

from common import AUGS, load_labels

from app.vj.queries import QUERIES, gate_questions, ground_questions
from app.vj.runtime import VideoJev


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--augs", default="none")
    p.add_argument("--head")
    p.add_argument("--hidden", help="папка для кэша скрытых состояний")
    p.add_argument("--only", help="через запятую: имена файлов")
    a = p.parse_args()

    import torch

    vj = VideoJev(a.model, a.head)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    hidden_dir = Path(a.hidden) if a.hidden else None
    if hidden_dir:
        hidden_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["file"], r["aug"]))
    videos = load_labels()
    if a.only:
        keep = set(a.only.split(","))
        videos = [v for v in videos if v["file"] in keep]
    questions = gate_questions() + ground_questions([q.code for q in QUERIES])

    with out.open("a") as f:
        for aug in a.augs.split(","):
            for v in videos:
                if (v["file"], aug) in done:
                    continue
                t0 = time.perf_counter()
                sample = vj.decode(str(Path(a.videos) / v["file"]), transform=AUGS[aug])
                for wi, (s, e) in enumerate(vj.windows(float(sample["sampled_duration"]))):
                    w = vj.prepare_window(sample, s, e)
                    res = vj.ask(w, questions, keep_hidden=bool(hidden_dir))
                    vj.free(w)
                    rec = dict(file=v["file"], aug=aug, start=s, end=e,
                               gate={r.id: r.probabilities for r in res if r.task == "choice"},
                               ground={r.id: dict(times=r.times, probs=r.probabilities)
                                       for r in res if r.task == "grounding"})
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    if hidden_dir:
                        torch.save({"file": v["file"], "aug": aug, "start": s, "end": e,
                                "hidden": {f"{r.task}:{r.id}": r.hidden for r in res},
                                "ground_times": {r.id: r.times for r in res if r.task == "grounding"}},
                                   hidden_dir / f"{Path(v['file']).stem}__{aug}__{wi}.pt")
                print(f"{v['file']} {aug} {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
