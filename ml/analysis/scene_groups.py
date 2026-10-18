"""Camera-session groups and leakage indicators for the organizer's dataset.

For every video: a static-background signature (median over 24 frames sampled across the clip,
downscaled to 24x44 (w x h) grayscale, z-normalised), mean brightness, motion energy (mean absolute frame
difference), and camera drift (1 - correlation of the first-six-frame and last-six-frame backgrounds).
Videos whose signatures correlate at >= --threshold are joined into one setup (union-find). The
groups feed analysis/setups.json and leave-one-setup-out evaluation (ml/evalkit.py).

    python ml/analysis/scene_groups.py --videos open/videos --out analysis/setups.json --threshold 0.92
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sample_frames(path, n=24):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    for i in np.linspace(0, max(total - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    return frames, total / fps


def signature(frames, size=(24, 44)):  # (width, height) for cv2.resize: portrait frames
    med = np.median(np.stack([cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]), axis=0)
    v = med.astype(np.float32).ravel()
    return (v - v.mean()) / (v.std() + 1e-6)


def cluster(S, threshold):
    n = len(S)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] >= threshold:
                parent[find(i)] = find(j)
    roots, out = {}, []
    for i in range(n):
        r = find(i)
        roots.setdefault(r, len(roots))
        out.append(roots[r])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default=str(ROOT / "open" / "videos"))
    p.add_argument("--labels", default=str(ROOT / "analysis" / "labels.json"))
    p.add_argument("--out", help="write setups.json here (keeps an existing meta block)")
    p.add_argument("--threshold", type=float, default=0.92)
    a = p.parse_args()

    labels = {v["file"]: v for v in json.loads(Path(a.labels).read_text())["videos"]}
    rows, sigs = [], {}
    for path in sorted(Path(a.videos).glob("*.mov")):
        frames, dur = sample_frames(path)
        sigs[path.name] = signature(frames)
        first, last = signature(frames[:6]), signature(frames[-6:])
        small = [cv2.resize(f, (58, 106), interpolation=cv2.INTER_AREA).astype(np.float32) for f in frames]
        lab = labels.get(path.name, {})
        ev = lab["events"][0] if lab.get("events") else {}
        rows.append(dict(file=path.name, duration=round(dur, 1), verdict=lab.get("verdict"), type=ev.get("type_code"),
                         brightness=round(float(np.mean([f.mean() for f in frames])), 1),
                         motion=round(float(np.mean([np.abs(x - y).mean() for x, y in zip(small[:-1], small[1:])])), 2),
                         cam_drift=round(float(1 - np.corrcoef(first, last)[0, 1]), 3)))
    names = [r["file"] for r in rows]
    S = np.array([[float(np.corrcoef(sigs[x], sigs[y])[0, 1]) for y in names] for x in names])
    for th in (0.97, 0.95, 0.92, 0.90, 0.85):
        c = cluster(S, th)
        print(f"threshold {th}: {len(set(c))} clusters, sizes {sorted(np.bincount(c).tolist(), reverse=True)}")
    groups = {}
    for r, c in zip(rows, cluster(S, a.threshold)):
        r["setup"] = f"S{c}"
        groups.setdefault(r["setup"], []).append(r["file"])
    print(f"\n{'file':10} {'dur':>6} {'verdict':12} {'type':24} {'bright':>6} {'motion':>6} {'drift':>6} setup")
    for r in sorted(rows, key=lambda r: (r["setup"], r["file"])):
        print(f"{r['file']:10} {r['duration']:6} {str(r['verdict']):12} {str(r['type']):24} {r['brightness']:6} "
              f"{r['motion']:6} {r['cam_drift']:6} {r['setup']}")
    if a.out:
        out = Path(a.out)
        meta = json.loads(out.read_text()).get("meta", {}) if out.exists() else {}
        out.write_text(json.dumps({"meta": meta, "setups": groups}, ensure_ascii=False, indent=2))
        print("wrote", out)


if __name__ == "__main__":
    main()
