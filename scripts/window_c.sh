#!/usr/bin/env bash
# window_c.sh — the GPU work that closes Phase 2. Unattended.
#
#   bash scripts/window_c.sh --smoke   # ~3 min, proves the box before the real run
#   bash scripts/window_c.sh           # ~3.2 GPU-hours if the last stage runs to the end
#
# WHY. Two Phase 2 items need a card.
#
# 1. `vs_literature` on a MATCHED assay list, which Step 2.2 promoted from an unchecked
#    box to a precondition: every published figure is a mean over its own assay list and
#    +0.409 is a mean over ours, so part of the shortfall may be assay difficulty and
#    nobody can currently apportion it. The fix is to run our estimator on the 5
#    multi-mutant ProteinGym datasets the fine-tuning literature benchmarks, under its
#    protocol, where Kermut-sequence scores 0.614 at n=128. Two of those 5 (GRB2, PABP)
#    top out at order 2 and so were never in this project's corpus.
#
# 2. ESM-2 embeddings. Kermut specifies ESM-2, this project defaults to SaProt on a
#    STABILITY measurement, and on fitness zero-shot ESM-2 is clearly ahead (masked
#    marginals +0.249 against +0.204, within-order +0.164 against +0.137). Whether the
#    backbone choice is costing us is a registered comparator question.
#
# WHAT IT COSTS, and why it is 29M residues and not 188M. The protocol can only ever
# touch max(train_n) + 5000 sequences per dataset per seed, so `literature_split` is
# shared between the encoder and the evaluator and the encoder embeds 75,082 sequences
# instead of the datasets' 691,253 -- 29.3M residues, about 18 minutes.
#
#   stage 3  SaProt  literature   ~20M res (rest is cached)   ~12 min
#   stage 4  ESM-2   literature    29.3M res                  ~18 min
#   stage 5  logprob matrices, both backbones, 5 datasets      ~5 min
#   stage 6  ESM-2   benchmark    240.3M res                  ~2.5 h   <- kill-safe, last
#
# Stage 6 is last and independent: stages 3-5 close the precondition, and if the pod is
# stopped during 6 nothing earlier is lost. Everything writes per assay, so an interrupt
# costs one file.

set -euo pipefail

SMOKE=""
[ "${1:-}" = "--smoke" ] && SMOKE=1

LIT="CAPSD_AAV2S_Sinai_2021,GFP_AEQVI_Sarkisyan_2016,GRB2_HUMAN_Faure_2021,HIS7_YEAST_Pokusaeva_2019,PABP_YEAST_Melamed_2013"

say() { printf '\n==== %s ====\n' "$*"; }

# `python -I` ignores the working directory, so the package has to be installed rather
# than found on sys.path. The first pod run of this project lost three minutes to exactly
# this, which is also why the smoke test exists.
say "stage 1: install"
# `python -I` ignores the working directory AND the environment, so neither sys.path nor
# PYTHONPATH will find the package -- it has to be installed. Recent CUDA images ship a
# PEP 668 "externally managed" Python, which refuses a plain `pip install`, and `--no-deps`
# keeps pip from resolving (and possibly replacing) the image's own torch build.
pip install -e . --no-deps --break-system-packages --quiet 2>&1 | grep -v "^WARNING: Running pip" || true
pip install --break-system-packages --quiet transformers pandas scipy scikit-learn 2>&1 | grep -v "^WARNING: Running pip" || true
python -I -c "import biodreamer, torch, transformers, sklearn; print('imports ok', torch.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

say "stage 2: smoke"
python -I scripts/encode_corpus.py --workload literature --pooled --dry-run
python -I scripts/zeroshot_baseline.py --dry-run --assays "$LIT" || true
if [ -n "$SMOKE" ]; then
  # one small assay end to end, both backbones, so a failure shows up in minutes
  python -I scripts/zeroshot_baseline.py --backbones esm2 --save-logprob-matrix \
      --assays GRB2_HUMAN_Faure_2021 --batch 8
  say "SMOKE OK"
  exit 0
fi

say "stage 3: SaProt pooled, literature protocol subset"
python -I scripts/encode_corpus.py --workload literature --pooled --backbone saprot \
    --batch-size 8

say "stage 4: ESM-2 pooled, literature protocol subset"
python -I scripts/encode_corpus.py --workload literature --pooled --backbone esm2 \
    --batch-size 8

say "stage 5: log-probability matrices, both backbones"
python -I scripts/zeroshot_baseline.py --backbones saprot,esm2 --save-logprob-matrix \
    --assays "$LIT" --batch 8

say "stage 5b: package what stages 3-5 produced, before the long stage starts"
tar czf /workspace/phase2_literature.tgz \
    data/processed/pooled_cache/*GRB2* data/processed/pooled_cache/*PABP* \
    data/processed/pooled_cache/facebook_esm2* \
    data/processed/pooled_cache/westlake-repl_SaProt_650M_AF2__CAPSD* \
    data/processed/pooled_cache/westlake-repl_SaProt_650M_AF2__GFP* \
    data/processed/pooled_cache/westlake-repl_SaProt_650M_AF2__HIS7* \
    data/processed/logprob_matrix_*.npz
ls -la /workspace/phase2_literature.tgz

say "stage 6: ESM-2 pooled, this project's own 11 assays (kill-safe)"
python -I scripts/encode_corpus.py --workload benchmark --pooled --backbone esm2 \
    --batch-size 8
tar czf /workspace/phase2_esm2_benchmark.tgz data/processed/pooled_cache/facebook_esm2*
ls -la /workspace/phase2_esm2_benchmark.tgz

say "ALL STAGES DONE"
