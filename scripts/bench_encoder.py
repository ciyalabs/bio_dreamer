"""Encoder throughput and cache-hit benchmark — Step 0.4 `cache_throughput`.

Answers the two questions the step gates on: how fast can we encode, and does
the cache actually pay for itself. Also prints the storage budget, which is the
number that decides whether a planned cache fill is feasible at all.

    python -I scripts/bench_encoder.py --backbone saprot --n 64
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from biodreamer.protein_dreamer.cache import Cache, estimate_bytes, format_bytes
from biodreamer.protein_dreamer.encoder import ProteinEncoder

# GB1 wild type: the shortest real held-out landscape, so the cold pass is quick
GB1 = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
AAS = "ACDEFGHIKLMNPQRSTVWY"


def variants(wt: str, n: int, seed: int = 0) -> list[str]:
    """`n` distinct single mutants of `wt`."""
    rng = np.random.default_rng(seed)
    out, seen = [], {wt}
    while len(out) < n:
        i = int(rng.integers(len(wt)))
        a = AAS[int(rng.integers(20))]
        if a == wt[i]:
            continue
        v = wt[:i] + a + wt[i + 1 :]
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="saprot", choices=["saprot", "esm2"])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cache-root", default="models/cache_bench")
    ap.add_argument("--out", default="data/processed/encoder_bench.json")
    a = ap.parse_args()

    seqs = variants(GB1, a.n)
    cache = Cache(root=a.cache_root)
    cache.clear()
    enc = ProteinEncoder(a.backbone, cache=cache)

    # warm the weights so the measurement is encode time, not model load
    t0 = time.perf_counter()
    enc.encode([GB1], use_cache=False)
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = enc.encode(seqs, batch_size=a.batch_size)
    cold_s = time.perf_counter() - t0
    cold_hits, cold_misses = cache.stats.hits, cache.stats.misses

    t0 = time.perf_counter()
    enc.encode(seqs, batch_size=a.batch_size)
    warm_s = time.perf_counter() - t0

    L, d = out[0].shape
    res = {
        "backbone": enc.name,
        "n_sequences": a.n,
        "length": L,
        "dim": d,
        "batch_size": a.batch_size,
        "model_load_s": round(load_s, 2),
        "cold_s": round(cold_s, 3),
        "warm_s": round(warm_s, 3),
        "cold_seq_per_s": round(a.n / cold_s, 1),
        "warm_seq_per_s": round(a.n / warm_s, 1),
        "speedup": round(cold_s / warm_s, 1) if warm_s else None,
        "cold_hits": cold_hits,
        "cold_misses": cold_misses,
        "cache_entries": cache.count(),
        "cache_bytes": cache.size_bytes(),
        "bytes_per_sequence": cache.size_bytes() // max(cache.count(), 1),
    }

    print(f"  backbone        {res['backbone']}")
    print(f"  model load      {res['model_load_s']} s (excluded from timings)")
    print(f"  cold pass       {res['cold_s']} s  -> {res['cold_seq_per_s']} seq/s")
    print(f"  warm pass       {res['warm_s']} s  -> {res['warm_seq_per_s']} seq/s")
    print(f"  cache speedup   {res['speedup']}x")
    print(f"  per sequence    {format_bytes(res['bytes_per_sequence'])}  (L={L}, d={d})")

    print("\n  storage budget for planned workloads:")
    print(
        cache.budget_report(
            {
                "GB1 doubles (raw d)": (536_000, 56, d),
                "GB1 doubles (adapter d=128)": (536_000, 56, 128),
                "TrpB4 (raw d)": (159_129, 390, d),
                "TrpB4 (adapter d=128)": (159_129, 390, 128),
                "ProteinGym all (raw, L~300)": (2_465_767, 300, d),
            }
        )
    )
    print(
        "\n  Cache Z_t after the adapter, not the raw backbone output: at the raw "
        f"dim ({d}) ProteinGym alone would need "
        f"{format_bytes(estimate_bytes(2_465_767, 300, d))}."
    )

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
