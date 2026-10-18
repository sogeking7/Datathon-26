#!/usr/bin/env bash
# Experiments from reports/experiment_plan.md, in order. Two tiers:
#   bash ml/run_experiments.sh offline   # tier A: cached scores / hidden states only (laptop or server CPU)
#   bash ml/run_experiments.sh gpu       # tier B: new VideoJev passes on the T4 (prints the commands, runs scoring)
# Inputs (paths as in ml/README.md §4; copy them from the server with scripts/fetch_runs.sh):
#   runs/full/scores.jsonl      zero-shot gate/grounding scores for augs none,hflip,persp
#   runs/full/cache/*.pt        hidden states for train_head.py
#   runs/head/scores_cv.jsonl   LOVO scores of the production head (ml/train_head.py output)
# Every run appends one row to reports/experiments.csv; predictions go to runs/pred/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-service/.venv/bin/python}
MODEL=${MODEL:-/models/videojev-4b-fp16}
REG=reports/experiments.csv
mkdir -p runs/pred

need() { [ -e "$1" ] || { echo "missing $1 (see header of this script)"; exit 1; }; }
exp() { "$PY" ml/experiment.py --registry "$REG" "$@"; }

offline() {
  need runs/head/scores_cv.jsonl
  H=runs/head/scores_cv.jsonl
  # E00 baseline with intervals (production head, LOVO) and the zero-shot reference
  exp --scores $H --exp-id E00 --hypothesis "baseline: head + F2 thresholds, nested LOVO" --save-pred runs/pred/E00.jsonl --decision keep
  [ -e runs/full/scores.jsonl ] && exp --scores runs/full/scores.jsonl --exp-id E00z --hypothesis "zero-shot gates, nested LOVO" --save-pred runs/pred/E00z.jsonl
  # E01 what the deployed service actually did before the confidence fix (audit F1)
  exp --scores $H --exp-id E01a --hypothesis "production thresholds.json, no filter" --fixed-thresholds ml/artifacts/thresholds.json --save-pred runs/pred/E01a.jsonl
  exp --scores $H --exp-id E01b --hypothesis "production thresholds.json + analyzer MIN_CONFIDENCE 0.5 filter (deployed rule before fix)" --fixed-thresholds ml/artifacts/thresholds.json --simulate-min-confidence 0.5 --compare runs/pred/E01a.jsonl
  # E03 aggregation, E04 calibration, E05 thresholds — all paired against E00
  exp --scores $H --exp-id E03a --hypothesis "top-2 mean over windows" --agg top2 --compare runs/pred/E00.jsonl --save-pred runs/pred/E03a.jsonl
  exp --scores $H --exp-id E03b --hypothesis "length-invariant Sidak p-value aggregation (Gaussian tail)" --agg zsidak --compare runs/pred/E00.jsonl --save-pred runs/pred/E03b.jsonl
  exp --scores $H --exp-id E04a --hypothesis "per-type bias subtraction on normal windows" --calib bias --compare runs/pred/E00.jsonl --save-pred runs/pred/E04a.jsonl
  exp --scores $H --exp-id E04b --hypothesis "bias subtraction + zsidak" --calib bias --agg zsidak --compare runs/pred/E00.jsonl --save-pred runs/pred/E04b.jsonl
  exp --scores $H --exp-id E05a --hypothesis "single global threshold" --no-per-type --compare runs/pred/E00.jsonl --save-pred runs/pred/E05a.jsonl
  exp --scores $H --exp-id E05b --hypothesis "per-type corrections shrunk to ±0.1" --shrink 0.1 --compare runs/pred/E00.jsonl --save-pred runs/pred/E05b.jsonl
  exp --scores $H --exp-id E05c --hypothesis "max recall at FP<=30%" --objective recall_at_fp --alpha 0.3 --compare runs/pred/E00.jsonl --save-pred runs/pred/E05c.jsonl
  # E06 generalization gap: leave-one-setup-out and repeated group 5-fold for baseline and candidates
  for id in E00 E03b E04a E05b; do
    case $id in E00) A="";; E03b) A="--agg zsidak";; E04a) A="--calib bias";; E05b) A="--shrink 0.1";; esac
    exp --scores $H --exp-id ${id}-setup --hypothesis "$id on leave-one-setup-out" --split setup $A
    exp --scores $H --exp-id ${id}-rgkf --hypothesis "$id on repeated group 5-fold x10" --split rgkf --k 5 --repeats 10 $A
  done
  # E08 two events per video
  exp --scores $H --exp-id E08 --hypothesis "up to two events of different types" --max-events 2 --compare runs/pred/E00.jsonl
  # E10 open-set probe with the existing gates
  exp --scores $H --exp-id E10 --hypothesis "leave-one-type-out: do sibling gates flag the held-out type?" --open-set
  # E02 / E09 head retrains (torch + cache; --device cpu works, ~minutes per fold on a laptop)
  if [ -d runs/full/cache ] && "$PY" -c "import torch" 2>/dev/null; then
    DEV=$("$PY" -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")
    "$PY" ml/train_head.py --cache runs/full/cache --model "$MODEL" --scores runs/full/scores.jsonl --out runs/head_alt --alt-policy ignore --device $DEV
    exp --scores runs/head_alt/scores_cv.jsonl --exp-id E02a --hypothesis "alt-type windows masked in head training" --compare runs/pred/E00.jsonl --save-pred runs/pred/E02a.jsonl
    exp --scores runs/head_alt/scores_cv.jsonl --exp-id E02a-setup --hypothesis "E02a leave-one-setup-out" --split setup
    "$PY" ml/train_head.py --cache runs/full/cache --model "$MODEL" --scores runs/full/scores.jsonl --out runs/head_altpos --alt-policy positive --device $DEV
    exp --scores runs/head_altpos/scores_cv.jsonl --exp-id E02b --hypothesis "alt-type windows as positives" --compare runs/pred/E00.jsonl
    "$PY" ml/train_head.py --cache runs/full/cache --model "$MODEL" --scores runs/full/scores.jsonl --out runs/head_soup --seeds 5 --ensemble soup --feature-noise 0.05 --device $DEV
    exp --scores runs/head_soup/scores_cv.jsonl --exp-id E09 --hypothesis "5-seed weight averaging + feature noise" --compare runs/pred/E00.jsonl
    "$PY" ml/train_head.py --cache runs/full/cache --model "$MODEL" --scores runs/full/scores.jsonl --out runs/head_setupcv --split setup --device $DEV
    exp --scores runs/head_setupcv/scores_cv.jsonl --exp-id E06h --hypothesis "head trained without the test session (true leave-one-setup-out)" --split setup
  else
    echo "skip E02/E09/E06h: need runs/full/cache and torch"
  fi
}

gpu() {
  need "$MODEL"
  echo "# E14 robustness: re-score the dataset under label-preserving transforms (≈25 min each on a T4)"
  echo "$PY ml/score_dataset.py --videos open/videos --model $MODEL --out runs/robust/scores.jsonl --augs none,hflip,rot_zoom,persp,flip_persp"
  echo "# then compare verdict flips per aug: for a in hflip rot_zoom persp flip_persp; do $PY ml/experiment.py --scores runs/robust/scores.jsonl --aug \$a --fixed-thresholds ml/artifacts/thresholds.json --exp-id E14-\$a --compare runs/pred/E01a.jsonl; done"
  echo "# resolution/compression/blur/brightness variants: see reports/generalization_report.md §3.2 (ffmpeg recipes), then score each folder with --videos <folder>"
  echo "# E11 generic queries: add TypeQuery entries in service/app/vj/queries.py, re-run score_dataset.py with --hidden, evaluate with --open-set"
  echo "# E12 order sensitivity: score twice with permuted options / shuffled gate order and compare gate probabilities"
  echo "# E13 coarse-to-fine timecode: implement the fine pass in app/vj/detector.py, evaluate tc_within_1s/2s/5s on E00 predictions"
  echo "# E15/E16 latency: time each stage in detector.run (log line) for Х-14 and a 20 s clip; WORKERS=1 vs 2 with two concurrent uploads"
}

case "${1:-offline}" in
  offline) offline ;;
  gpu) gpu ;;
  *) echo "usage: $0 offline|gpu"; exit 2 ;;
esac
