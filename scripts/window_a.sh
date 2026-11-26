#!/usr/bin/env bash
# window_a.sh — every GPU job Phase 1 needs, unattended, in one pass.
#
# RENTAL_PLAN.md calls this Window A. Nothing here needs a human watching it, which
# is the point: the rented box bills by availability, not by GPU utilisation, so the
# cheapest run is the one nobody has to babysit. Measured utilisation across the
# whole study is 6%.
#
#   bash scripts/window_a.sh            # the real thing
#   bash scripts/window_a.sh --dry-run  # prove every code path, no model, no GPU
#
# Order matters. The benchmark runs first because every cost estimate depends on it
# and it takes seconds. The probe encode runs second because it is the cheapest way
# to find out the cache and the backbone work before committing to a long job.
#
# What is NOT here, and why:
#   - encode_corpus.py --workload landscapes
#       The SSMuLA landscape CSVs store only the variable residues (4 characters for
#       TrpB4), not full protein sequences, and only TEV ships a "Complete Sequence"
#       column. Encoding 4-residue fragments through a protein language model is
#       meaningless. The variable-position mapping for TrpB4, GB1, ParD2 and ParD3
#       has to be established first -- see DATA_REGISTER.md. That is Window B work
#       and it needs no GPU, so it must not consume rented hours.
#   - anything at d=128
#       The 10x storage saving needs the trained adapter, which does not exist until
#       Phase 2. Window A caches raw d=1280; at this scope that is 152 GiB and fits.

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

# 1. Measure the card. Everything in RENTAL_PLAN.md scales with this number, and
#    the 30 TFLOP/s in cost_model.py is an estimate from the card's power budget.
run "1/5  benchmark the card" python -I scripts/bench_encoder.py

# 2. Smallest possible encode: proves backbone, cache and disk before a long run.
run "2/5  probe encode (200 sequences)" \
  python -I scripts/encode_corpus.py --workload probe

# 3. H1's real scope: 7 unbanned assays with order>=3. 287k sequences, 64M residues,
#    0.083 EFLOP. The cost model's 0.90 EFLOP assumed the whole corpus; the firewall
#    bans 49% of it and H1's protocol touches 2.6% of what remains.
run "3/5  encode H1 train+test" \
  python -I scripts/encode_corpus.py --workload h1

# 4. D1 folding. Wild types are folded twice to establish the predictor's own
#    reproducibility as the floor -- without that, "the structure does not move" is
#    unfalsifiable. Selection already happened on CPU; this only consumes a manifest.
run "4/5  D1 fold manifest" python -I scripts/d1_fold.py $DRY

# 5. D2 latent capacity (step 1.3). Decides per-residue versus pooled, and fixes d.
if [[ -f scripts/d2_latent_capacity.py ]]; then
  run "5/5  D2 latent capacity" python -I scripts/d2_latent_capacity.py $DRY
else
  printf '\n\033[1m== 5/5  D2 latent capacity -- SCRIPT NOT WRITTEN\033[0m\n' | tee -a "$LOG"
  echo "scripts/d2_latent_capacity.py does not exist yet. Write it BEFORE ordering," \
       "or Window A finishes early and the box bills while you write it." | tee -a "$LOG"
fi

printf '\nWindow A finished %s\n' "$(date -Is)" | tee -a "$LOG"
bdtrack status 2>&1 | head -30 | tee -a "$LOG"

cat <<'EOF' | tee -a "$LOG"

BEFORE CANCELLING THE SERVER
  1. rsync the cache home. It is the only expensive-to-recreate artifact, and the
     monthly disk goes away with the order:
       rsync -az --info=progress2 root@BOX:/root/bio_dreamer/models/cache_bench/ ./
  2. rsync data/processed/ back (fold PDBs, d1_structure.json, encoding_log.json).
  3. Commit and push. The tracker flags runs from a dirty tree as non-reproducible,
     and these are headline numbers.
  4. Re-read the cancel rule in RENTAL_PLAN.md: the EUR 209 setup fee is worth 610
     hours of runtime, so cancel only if the next GPU job is more than ~4 weeks out.
EOF
