#!/usr/bin/env bash
# window_b.sh — the one GPU job H1's failure calls for. Unattended, ~0.7 GPU-hours.
#
#   bash scripts/window_b.sh            # the real thing
#   bash scripts/window_b.sh --smoke    # two 93-residue assays at cap 200, ~5 min
#
# WHY. H1 failed at spearman_mean +0.409 against 0.55, and a plain
# RidgeCV(pooled_embedding ++ mutation_count) scored +0.390 -- the trained 266k head
# beat it by +0.019 and lost on median. The diagnosis in results/reward_benchmark.md
# acquits adapter capacity, the loss and per-assay normalisation, and lands on the
# representation: every configuration reads the same 1280-d MEAN-POOLED vector, which
# cannot express which position changed.
#
# WHAT IT COSTS, and why it is not the 170 GiB first estimated. Pooling only ever saved
# STORAGE; the forward pass is identical either way, so per-residue encoding costs the
# same GPU time as pooled did. And the hypothesis is about the mutated positions, so
# this caches k x 1280 per variant (mean k ~ 5) plus one wild-type latent per assay
# rather than whole per-residue tensors. Test variants are capped at 30,000 per assay,
# which is 27.7% of the residues and keeps Spearman's standard error near 0.006.
#
#   66.3M residues at the measured 27,000 res/s = 0.68 h ~ $0.45
#   output to bring home: ~3 GB
#
# Everything is paired: the readout comparison uses one head, recipe, seed and test
# subsample, and only the input vector changes.

set -euo pipefail

CAP=30000
ONLY=""
if [[ "${1:-}" == "--smoke" ]]; then
  CAP=200
  ONLY="--only F7YBW8_MESOW_Ding_2023,F7YBW8_MESOW_Aakre_2015"
  echo "SMOKE: 2 short assays at cap ${CAP}; results are NOT informative about the"
  echo "       hypothesis, which is about long proteins (CAPSD 735, SPG1 448)."
fi

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

LOG="data/processed/window_b_$(date +%Y%m%d_%H%M%S).log"
mkdir -p data/processed

run() {
  printf '\n\033[1m== %s\033[0m\n' "$1" | tee -a "$LOG"
  shift
  if ! "$@" 2>&1 | tee -a "$LOG"; then
    printf '\033[31mFAILED: %s\033[0m\n' "$*" | tee -a "$LOG"
    exit 1
  fi
}

nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a "$LOG"

# shellcheck disable=SC2086
run "encode mutated-position latents (benchmark instance)" \
  python -I -W ignore scripts/encode_positional.py \
    --instance benchmark --test-cap "$CAP" --batch-size 16 $ONLY

run "compare readouts: pooled vs mut_mean vs displacement vs both" \
  python -I -W ignore scripts/positional_readout.py --instance benchmark

printf '\n\033[1mdone.\033[0m cache in data/processed/positional_cache/ — rsync it home,\n'
printf 'then terminate the pod. Log: %s\n' "$LOG"
du -sh data/processed/positional_cache/ | tee -a "$LOG"
