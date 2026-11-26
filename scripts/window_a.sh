#!/usr/bin/env bash
# window_a.sh — the GPU jobs Phase 1 needs. Unattended, one pass, ~5 GPU-hours.
#
#   bash scripts/window_a.sh            # the real thing
#   bash scripts/window_a.sh --dry-run  # prove every code path, no model, no GPU
#
# Window A produces MEASUREMENTS, not a cache. That is deliberate. Step 1.3 fixed the
# latent dimension at d=128, and the adapter that projects 1280 -> 128 does not exist
# until Phase 2. Caching raw d=1280 latents here would mean moving 152 GiB home, when
# re-encoding later costs 0.8 GPU-hours. So this script encodes only a sample, to
# measure throughput, and leaves the real cache for Phase 2.
#
# Everything it writes comes to under 100 MB.

set -euo pipefail

DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY="--dry-run"

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

LOG="data/processed/window_a_$(date +%Y%m%d_%H%M%S).log"
mkdir -p data/processed

run() {
  printf '\n\033[1m== %s\033[0m\n' "$1" | tee -a "$LOG"
  shift
  if ! "$@" 2>&1 | tee -a "$LOG"; then
    echo "FAILED: $*" | tee -a "$LOG"
    return 1
  fi
}

printf 'Window A starting %s\n' "$(date -Is)" | tee "$LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>&1 | tee -a "$LOG" || true

# 1. Measure the card. Every hour and euro in RENTAL_PLAN.md scales with this, and
#    the figure in cost_model.py is an estimate from the card's power budget.
run "1/4  benchmark the card" python -I scripts/bench_encoder.py

# 2. A 200-sequence encode. Proves the backbone, the cache and the disk work, and
#    gives a real residues-per-second number for Phase 2's planning. Not a cache fill.
run "2/4  encoding throughput sample" \
  python -I scripts/encode_corpus.py --workload probe

# 3. The reason this window exists. ESMFold does not fit a 4 GB laptop card.
#    Wild types are folded twice so the floor is the predictor's own reproducibility;
#    without that, "the structure does not move" cannot be falsified.
run "3/4  D1 fold the manifest" python -I scripts/d1_fold.py $DRY

# 4. Step 1.3 already ran at home and fixed d=128, but SaProt had no structure tokens
#    there, so it was compared as a plain sequence model. Now that the folds exist,
#    re-run it with real 3Di to settle the backbone choice.
# d2 has no --dry-run: it runs fine without a big GPU, so the dry form is a
# two-protein pass that exercises the same code.
if [[ -n "$DRY" ]]; then
  run "4/4  D2 re-run with real 3Di (dry: 2 proteins)" \
    python -I scripts/d2_latent_capacity.py --limit 2 --no-track
else
  run "4/4  D2 re-run with real 3Di" python -I scripts/d2_latent_capacity.py
fi

printf '\nWindow A finished %s\n' "$(date -Is)" | tee -a "$LOG"
bdtrack status 2>&1 | sed -n '/Phase 1/,/Phase 2/p' | tee -a "$LOG"
du -sh data/processed/d1_folds 2>/dev/null | tee -a "$LOG" || true

cat <<'EOF' | tee -a "$LOG"

BEFORE DESTROYING THE BOX
  Copy these home. Under 100 MB in total:
    data/processed/d1_folds/      the predicted structures
    data/processed/*.json         d1_structure, d2_latent, encoding_log
    experiments.db                the tracker ledger
    data/processed/window_a_*.log this log

  Do NOT copy models/ -- those are the raw d=1280 embedding cache and the downloaded
  weights. The cache is 152 GiB and re-encoding it costs 0.8 GPU-hours; the weights
  re-download from Hugging Face in minutes.

  Then commit and push. The tracker flags runs from a dirty tree as non-reproducible.
EOF
