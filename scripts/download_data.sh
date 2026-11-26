#!/usr/bin/env bash
# download_data.sh — fetch and VERIFY the datasets ProteinDreamer needs.
#
# This file used to be a stub: a long comment describing datasets for three
# research lines, with no download call anywhere in it. That is worse than having
# nothing, because provisioning a rented GPU box reads like it is handled.
#
# Sources and checksums come from DATA_REGISTER.md, which recorded what was actually
# audited. Every archive is checked after download and the script fails on mismatch
# rather than leaving a corrupt file for a later step to misread.
#
#   bash scripts/download_data.sh            # the three datasets Phase 1 needs
#   bash scripts/download_data.sh --verify   # check what is already on disk
#
# PREFER RSYNC WHEN PROVISIONING A RENTED BOX. scripts/provision_gpu_box.sh copies
# the archives from a machine that already has them, which transfers the exact bytes
# this project audited and cannot be affected by an upstream re-release. Use this
# script for a fresh clone with no local copy.
#
# MolDreamer and CellDreamer are future research lines with no acquired data; the
# old comment block listing their datasets described intent, not reality, and is gone.

set -euo pipefail
cd "$(dirname "$0")/.."

VERIFY_ONLY=0
[[ "${1:-}" == "--verify" ]] && VERIFY_ONLY=1

# name | destination | url | sha256 prefix (as recorded in DATA_REGISTER.md)
# ProteinGym is the training corpus; SSMuLA holds the combinatorial landscapes;
# Tsuboyama supplies measured ddG for steps 1.2, 1.3 and 2.1.
read -r -d '' DATASETS <<'EOF' || true
proteingym|data/raw/proteingym/DMS_ProteinGym_substitutions.zip|https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip|3a83766254ac
ssmula|data/raw/ssmula/data.zip|https://zenodo.org/records/13910506/files/data.zip|6acdf302d412
EOF

# Tsuboyama is a Zenodo record whose filename differs across versions, so it is
# handled separately and checked by size rather than a recorded prefix.
TSUBOYAMA_DEST="data/raw/tsuboyama/Processed_K50_dG_datasets.zip"
TSUBOYAMA_URL="https://zenodo.org/records/7992926/files/Processed_K50_dG_datasets.zip"
TSUBOYAMA_BYTES=1013678473

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

check_sha() {
  local path="$1" want="$2"
  local got
  got="$(sha256sum "$path" | cut -c1-12)"
  if [[ "$got" != "$want" ]]; then
    echo "  CHECKSUM MISMATCH for $path" >&2
    echo "    expected $want..., got $got..." >&2
    echo "    DATA_REGISTER.md records the audited bytes; upstream may have" >&2
    echo "    re-released. Do not use this file until the difference is understood." >&2
    return 1
  fi
  echo "  ok  $path  sha256:$got"
}

fetch() {
  local dest="$1" url="$2"
  mkdir -p "$(dirname "$dest")"
  if [[ -f "$dest" ]]; then
    echo "  already present: $dest"
    return 0
  fi
  if [[ "$VERIFY_ONLY" -eq 1 ]]; then
    echo "  MISSING: $dest"
    return 1
  fi
  echo "  downloading $url"
  curl -fL --retry 3 --retry-delay 5 -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

say "ProteinGym and SSMuLA"
rc=0
while IFS='|' read -r name dest url sha; do
  [[ -z "$name" ]] && continue
  echo "[$name]"
  fetch "$dest" "$url" || { rc=1; continue; }
  check_sha "$dest" "$sha" || rc=1
done <<< "$DATASETS"

say "Tsuboyama 2023"
if fetch "$TSUBOYAMA_DEST" "$TSUBOYAMA_URL"; then
  size="$(stat -c%s "$TSUBOYAMA_DEST")"
  if [[ "$size" -ne "$TSUBOYAMA_BYTES" ]]; then
    echo "  SIZE MISMATCH: expected $TSUBOYAMA_BYTES bytes, got $size" >&2
    rc=1
  else
    echo "  ok  $TSUBOYAMA_DEST  $size bytes"
  fi
else
  rc=1
fi

say "extract SSMuLA"
# The landscape CSVs are read directly from the extracted tree by
# biodreamer/eval/landscape.py, so this one must be unpacked; the other two are
# streamed from their archives and are left zipped.
if [[ -f data/raw/ssmula/data.zip && ! -d data/raw/ssmula/extracted/data ]]; then
  if [[ "$VERIFY_ONLY" -eq 0 ]]; then
    mkdir -p data/raw/ssmula/extracted
    unzip -q data/raw/ssmula/data.zip -d data/raw/ssmula/extracted
    echo "  extracted to data/raw/ssmula/extracted/"
  else
    echo "  MISSING: data/raw/ssmula/extracted/data"
    rc=1
  fi
else
  echo "  already extracted (or archive absent)"
fi

say "summary"
python3 - <<'PY'
from pathlib import Path
need = {
    "ProteinGym": "data/raw/proteingym/DMS_ProteinGym_substitutions.zip",
    "SSMuLA (archive)": "data/raw/ssmula/data.zip",
    "SSMuLA (extracted)": "data/raw/ssmula/extracted/data",
    "Tsuboyama": "data/raw/tsuboyama/Processed_K50_dG_datasets.zip",
}
for label, p in need.items():
    path = Path(p)
    mark = "ok " if path.exists() else "MISSING"
    size = ""
    if path.is_file():
        size = f"  {path.stat().st_size/1e6:,.0f} MB"
    print(f"  {mark:<8}{label:<22}{p}{size}")
PY

if [[ "$rc" -ne 0 ]]; then
  echo
  echo "one or more datasets are missing or failed verification" >&2
  exit 1
fi
echo
echo "Not fetched here, and not needed for Window A:"
echo "  FLIP2, SLIP, GB1 Wu 2016 supplement, HypercubeME clone -- see DATA_REGISTER.md"
echo "  model weights -- downloaded on first use by huggingface-hub"
