#!/usr/bin/env bash
# Copies the model-output artifacts from the GPU host to ./runs so that every tier-A experiment
# (ml/run_experiments.sh offline) runs on a laptop without a GPU.
#   scripts/fetch_runs.sh ubuntu@<host> <remote runs dir> [ssh key]
# Example: scripts/fetch_runs.sh ubuntu@3.83.152.128 /home/ubuntu/Datathon-26/runs ~/Downloads/vockey.pem
# Expected remote layout (ml/README.md §4): runs/full/scores.jsonl, runs/full/cache/*.pt, runs/head/scores_cv.jsonl,
# runs/head/head.pt, runs/head/thresholds.json. The cache is tens of MB (hidden states per window/question).
set -euo pipefail
HOST=${1:?user@host}; REMOTE=${2:?remote runs dir}; KEY=${3:-}
cd "$(dirname "$0")/.."
SSH="ssh -o ControlMaster=auto -o ControlPath=~/.ssh/cm-%r@%h:%p -o ControlPersist=600"
[ -n "$KEY" ] && SSH="$SSH -i $KEY"
mkdir -p runs
rsync -av --progress -e "$SSH" --include='*/' --include='*.jsonl' --include='*.json' --include='*.pt' --exclude='*' \
  "$HOST:$REMOTE/" runs/
echo "fetched into ./runs; next: bash ml/run_experiments.sh offline"
