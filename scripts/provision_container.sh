#!/usr/bin/env bash
# provision_container.sh — set up Window A on a rented GPU container.
#
# For RunPod, Vast.ai, or anything else that hands you a PyTorch container with the
# driver and CUDA already working. For Hetzner bare metal use provision_gpu_box.sh
# instead, which also installs the driver.
#
# Window A needs almost nothing: a 15 MB repo, a 236 KB manifest, and model weights
# that download from Hugging Face. No datasets. So this takes minutes, not hours.
#
#   bash provision_container.sh --from-host you@laptop:/path/to/bio_dreamer
#
# Or, if the repo is already on the box:
#   bash provision_container.sh

set -euo pipefail

FROM_HOST=""
WORKDIR="${WORKDIR:-/workspace/bio_dreamer}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-host) FROM_HOST="$2"; shift 2 ;;
    --workdir)   WORKDIR="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "1/5  check the GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA; pick a GPU container, not a CPU one"
p = torch.cuda.get_device_properties(0)
gb = p.total_memory / 1e9
print(f"{p.name}  {gb:.1f} GB")
assert gb > 20, f"ESMFold needs ~16 GB of headroom; this card has {gb:.1f} GB"
PY

say "2/5  base packages"
apt-get update -qq
apt-get install -y -qq git curl rsync unzip >/dev/null

say "3/5  repository"
if [[ -n "$FROM_HOST" ]]; then
  mkdir -p "$WORKDIR"
  # Only what Window A reads. The datasets stay home -- nothing here needs them.
  rsync -az --info=progress2 \
    --exclude '.venv/' --exclude '.git/' --exclude 'models/' --exclude 'tools/' \
    --exclude 'data/raw/' --exclude 'data/processed/hypercubeme/' \
    --exclude 'data/processed/d1_folds/' --exclude '__pycache__/' \
    "$FROM_HOST"/ "$WORKDIR"/
fi
cd "$WORKDIR"
test -f data/processed/d1_fold_manifest.json \
  || { echo "manifest missing -- run scripts/d1_build_manifest.py at home first" >&2; exit 1; }

say "4/5  python environment"
curl -fsSL https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e ".[ml,dev]"

say "5/5  foldseek"
mkdir -p tools
curl -fsSL https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz -o tools/fs.tar.gz
tar -xzf tools/fs.tar.gz -C tools/
tools/foldseek/bin/foldseek version

say "ready"
cat <<'EOF'
Run, in this order:

  python -I scripts/bench_encoder.py        # measure the card; takes a minute
  bash scripts/window_a.sh                  # the real work, unattended

When it finishes, copy the results home. It is under 100 MB:

  rsync -az root@BOX:/workspace/bio_dreamer/data/processed/d1_folds/ ./data/processed/d1_folds/
  rsync -az root@BOX:/workspace/bio_dreamer/data/processed/*.json ./data/processed/
  rsync -az root@BOX:/workspace/bio_dreamer/experiments.db ./

Then destroy the container. Nothing on it is expensive to recreate.
EOF
