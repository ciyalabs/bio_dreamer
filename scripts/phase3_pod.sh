#!/usr/bin/env bash
# phase3_pod.sh — Step 3.1 + 3.2 on a rented GPU, unattended.
#
# Encodes the FULL per-residue state for each planning assay's transition endpoints and
# trains World Model B on it, reading the amended Step 3.2 gate. Runs assay by assay,
# smallest first, and frees each cache once that assay is trained:
#
#   * peak disk is the LARGEST single assay (~52 GB on CAPSD), not the 119 GB total
#   * if the pod dies, every assay already finished is already reported
#   * re-training an assay needs re-encoding, which is minutes here and 48 hours at home
#
# Why rent at all: per-residue encoding measured 270 res/s on the project's local GTX
# 1650 against ~26,000 res/s on an A40. 46.5M residues is 48 hours there and ~30 minutes
# here. The cache is far too large for the ~10 MB/s link home, so the training that reads
# it has to be here too; only metrics and checkpoints go back.
#
#   bash scripts/phase3_pod.sh                 # all seven, smallest first
#   bash scripts/phase3_pod.sh --keep-cache    # keep each cache (needs ~130 GB)
#   bash scripts/phase3_pod.sh --epochs 10

set -euo pipefail

EPOCHS=20
KEEP_CACHE=0
MAX_TRANSITIONS=60000
OUT_DIR="results/pod_phase3"
PY="${PY:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs)          EPOCHS="$2"; shift 2 ;;
    --keep-cache)      KEEP_CACHE=1; shift ;;
    --max-transitions) MAX_TRANSITIONS="$2"; shift 2 ;;
    --out)             OUT_DIR="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# smallest first: the cheap assays confirm the pipeline before CAPSD commits 20 minutes
# of GPU, and each one's result is written before the next begins
ASSAYS=(
  GCN4_YEAST_Staller_2018
  D7PM05_CLYGR_Somermeyer_2022
  GFP_AEQVI_Sarkisyan_2016
  Q6WV12_9MAXI_Somermeyer_2022
  Q8WTC7_9CNID_Somermeyer_2022
  PHOT_CHLRE_Chen_2023
  CAPSD_AAV2S_Sinai_2021
)

say "0/3  environment"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY - <<'PY'
import shutil
import torch
assert torch.cuda.is_available(), "no CUDA -- this needs a GPU pod"
p = torch.cuda.get_device_properties(0)
print(f"{p.name}  {p.total_memory/1e9:.1f} GB VRAM")
free = shutil.disk_usage(".").free / 1e9
print(f"{free:.0f} GB free on the working volume")
# the largest single assay is 52.3 GB; leave room for weights and the next encode
assert free > 70, f"need >70 GB free for the per-residue cache, have {free:.0f} GB"
PY
mkdir -p "$OUT_DIR"

say "1/3  sanity: scope and cost, nothing encoded"
$PY -I scripts/encode_residue_states.py --instance planning \
    --max-transitions "$MAX_TRANSITIONS" --dry-run | tee "$OUT_DIR/scope.txt"

for A in "${ASSAYS[@]}"; do
  say "2/3  $A — encode"
  # batch by length: CAPSD is 735 residues, so a batch that suits 118 would blow up
  # sized for a 46 GB A40 at fp16: roughly 24k residues per encode batch. The first
  # run used a quarter of these and left the card idle between batches.
  case "$A" in
    CAPSD_AAV2S_Sinai_2021) EB=32;  TB=24 ;;
    PHOT_CHLRE_Chen_2023)   EB=192; TB=128 ;;
    *)                      EB=96;  TB=64 ;;
  esac
  $PY -I scripts/encode_residue_states.py --assay "$A" \
      --max-transitions "$MAX_TRANSITIONS" --batch-size "$EB" \
      2>&1 | tee "$OUT_DIR/encode_$A.log"

  say "2/3  $A — train (per-residue, amended gate)"
  # --min-transitions 1000 is the registered floor; GCN4 has 144 and is expected to be
  # skipped and RECORDED as skipped rather than averaged into the gate
  $PY -I scripts/train_wm_b.py --state residue --assay "$A" \
      --epochs "$EPOCHS" --batch "$TB" --max-transitions "$MAX_TRANSITIONS" \
      --save-checkpoint "$OUT_DIR/checkpoints" \
      --out "$OUT_DIR/wm_residue_$A.json" \
      2>&1 | tee "$OUT_DIR/train_$A.log"

  if [[ "$KEEP_CACHE" -eq 0 ]]; then
    rm -f data/processed/residue_cache/*__"$A".npy \
          data/processed/residue_cache/*__"$A".done.npy
    echo "  freed the per-residue cache for $A"
  fi
  df -h . | tail -1
done

say "3/3  collect"
$PY -I scripts/collect_phase3.py --dir "$OUT_DIR" --out "$OUT_DIR/summary.json"
echo
echo "Bring home: $OUT_DIR  (metrics, logs and checkpoints only -- never the cache)"
du -sh "$OUT_DIR"
