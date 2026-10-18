"""Re-score cached hidden states with a given head for every augmentation in the cache (robustness, E14).

score_dataset.py --hidden stores, per window and question, the hidden states fed to the decision head; so any
head (pretrained, head_v1, a retrain) can be applied to the hflip/persp copies of every window without touching
the backbone. Output: scores.jsonl rows (file, aug, start, end, gate, ground) for the chosen augs.

    python ml/rescore_cache.py --cache runs/full/cache --head /models/head.pt --scores runs/full/scores.jsonl \
        --augs none,hflip,persp --out runs/head_v1/scores_augs.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from train_head import Head


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--head", required=True, help="head.pt state dict (train_head.py) or 'pretrained:<model_dir>'")
    p.add_argument("--scores", required=True, help="scores.jsonl of the same pass (for window order and grounding times)")
    p.add_argument("--augs", default="none,hflip,persp")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    head = Head().to(a.device)
    if a.head.startswith("pretrained:"):
        from train_head import pretrained_head

        head.load_state_dict(pretrained_head(a.head.split(":", 1)[1]))
    else:
        head.load_state_dict(torch.load(a.head, map_location=a.device))
    head.eval()
    counters = defaultdict(int)
    cache = Path(a.cache)
    augs = a.augs.split(",")
    n = 0
    with Path(a.out).open("w") as out:
        for line in Path(a.scores).read_text().splitlines():
            r = json.loads(line)
            if r["aug"] not in augs:
                continue
            wi = counters[(r["file"], r["aug"])]
            counters[(r["file"], r["aug"])] += 1
            f = cache / f"{Path(r['file']).stem}__{r['aug']}__{wi}.pt"
            if not f.exists():
                print("missing", f)
                continue
            d = torch.load(f)
            gate, ground = {}, {}
            for key, h in d["hidden"].items():
                task, code = key.split(":", 1)
                logits = head(h["pairs"].to(a.device), h["query"].to(a.device))
                if task == "choice":
                    gate[code] = logits.softmax(-1).cpu().tolist()
                else:
                    ground[code] = dict(times=r["ground"][code]["times"], probs=logits.sigmoid().cpu().tolist())
            out.write(json.dumps({**r, "gate": gate, "ground": ground}) + "\n")
            n += 1
    print("wrote", n, "rows to", a.out)


if __name__ == "__main__":
    main()
