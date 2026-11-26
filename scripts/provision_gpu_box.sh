#!/usr/bin/env bash
# provision_gpu_box.sh — turn a fresh Hetzner GEX bare-metal box into a working
# ProteinDreamer runner, in one pass, with verification at the end.
#
# WHY THIS EXISTS
#   A GEX server is unmanaged bare metal: the EUR 209 setup fee buys provisioned
#   hardware, not a configured environment. Billing starts when the server is
#   assigned to the account, not when you power it on, and powering off does not
#   pause it. Every minute spent working out the driver install is billed at the
#   same rate as a training run, so the install is scripted and the script is
#   tested before the order is placed.
#
# RUN THIS ON THE BOX, as root, after installimage has put Ubuntu 24.04 on it:
#
#   bash provision_gpu_box.sh --from-host you@your-laptop:/path/to/bio_dreamer
#
# The repo and the audited data come over rsync rather than being re-downloaded.
# That is deliberate: the checksums in DATA_REGISTER.md describe the exact bytes
# this project audited, scripts/download_data.sh is a stub with no download calls
# in it, and re-fetching from upstream would mean re-auditing provenance on a
# billing machine.
#
# WHAT IT DOES NOT DO
#   It does not handle git credentials, and it does not fetch anything from a
#   private repository. Pass --from-host and it pulls over your own ssh.

set -euo pipefail

FROM_HOST=""
SKIP_DRIVER=0
WORKDIR="/root/bio_dreamer"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-host) FROM_HOST="$2"; shift 2 ;;
    --workdir)   WORKDIR="$2"; shift 2 ;;
    --skip-driver) SKIP_DRIVER=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "1/7  base packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
  build-essential git curl rsync unzip pkg-config \
  python3-dev python3-venv \
  hmmer mmseqs2
# hmmer and mmseqs2 are the non-pip tools DATA_REGISTER.md records as required.
# foldseek is a static binary, fetched below -- it is not in apt.

if [[ "$SKIP_DRIVER" -eq 0 ]]; then
  say "2/7  NVIDIA driver"
  # ubuntu-drivers picks the recommended branch for the installed card, which is
  # what we want: an RTX PRO 4000 Blackwell needs a recent driver, and pinning a
  # version here would rot.
  apt-get install -y -qq ubuntu-drivers-common
  ubuntu-drivers autoinstall
  echo "driver installed -- a REBOOT is required before the GPU is usable"
  echo "reboot now, then re-run with --skip-driver"
  exit 0
else
  say "2/7  NVIDIA driver (skipped)"
  nvidia-smi || { echo "nvidia-smi failed: driver not active, reboot first" >&2; exit 1; }
fi

say "3/7  uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -fsSL https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

say "4/7  repository and data"
if [[ -n "$FROM_HOST" ]]; then
  mkdir -p "$WORKDIR"
  # --exclude the things that are large and regenerable; include the manifests
  # and the audited archives, which are what the box actually needs.
  rsync -az --info=progress2 \
    --exclude '.venv/' --exclude '.git/' --exclude 'models/' \
    --exclude 'data/processed/hypercubeme/' --exclude 'data/processed/d1_folds/' \
    --exclude '__pycache__/' --exclude '*.pyc' \
    "$FROM_HOST"/ "$WORKDIR"/
else
  echo "no --from-host given; expecting $WORKDIR to exist already"
fi
cd "$WORKDIR"
test -f pyproject.toml || { echo "no pyproject.toml in $WORKDIR" >&2; exit 1; }

say "5/7  python environment"
uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e ".[ml,dev]"

say "6/7  foldseek"
mkdir -p tools
if [[ ! -x tools/foldseek/bin/foldseek ]]; then
  curl -fsSL https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz \
    -o tools/foldseek-linux-avx2.tar.gz
  tar -xzf tools/foldseek-linux-avx2.tar.gz -C tools/
fi
tools/foldseek/bin/foldseek version

say "7/7  verification"
python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  {p.name}  {p.total_memory/1e9:.1f} GB  capability {p.major}.{p.minor}")
    assert p.total_memory > 20e9, "expected >=24 GB; ESMFold needs ~16 GB"
PY

echo
echo "running the test suite (no GPU required for most of it)"
python -m pytest -q

echo
echo "FIRST THING TO DO, before any long job:"
echo "  python -I scripts/bench_encoder.py"
echo "  -> then replace the ESTIMATED 30 TFLOP/s for hetzner-gex45 in"
echo "     scripts/cost_model.py with the measured number. Every hour and euro"
echo "     in RENTAL_PLAN.md scales with it, and step 0.4 did exactly this for"
echo "     the laptop."
echo
echo "then:  bash scripts/window_a.sh"
