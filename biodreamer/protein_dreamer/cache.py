"""Per-residue embedding cache — Step 0.4 of RESEARCH_PLAN.md.

Every later phase reads embeddings through this. Without a cache, repeated
backbone forward passes dominate runtime: a single pass over ProteinGym's
2.4M variants at ~100 seq/s is roughly seven hours, and the diagnostics in
Phase 1 need several such passes.

Entries are keyed by `(backbone, sequence_hash)` so two backbones never collide
and an unchanged sequence is never re-encoded. Values are stored as raw `.npy`
under a two-level fan-out, which keeps directory sizes manageable and makes an
entry trivially inspectable from the shell.

Storage is the thing that bites. Per-residue embeddings scale with sequence
length, so budget before filling:

    GB1    L=56   d=1280  fp16  ->  140 KiB/seq  ->  536k doubles =  72 GiB
    GB1    L=56   d=128   fp16  ->   14 KiB/seq  ->  536k doubles =  7.2 GiB

`estimate_bytes` does this arithmetic, and `Cache.budget_report` prints it for a
planned workload. The adapter projects 1280 -> d before anything is cached, so
cache `Z_t`, not the raw backbone output.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path("models/cache")


def sequence_hash(sequence: str) -> str:
    """Stable key for a sequence, independent of case and whitespace."""
    return hashlib.sha256(sequence.strip().upper().encode()).hexdigest()[:20]


def estimate_bytes(n_sequences: int, length: int, dim: int, dtype: str = "float16") -> int:
    """Bytes needed to cache `n_sequences` per-residue embeddings."""
    itemsize = np.dtype(dtype).itemsize
    return n_sequences * length * dim * itemsize


def format_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    evictions: int = 0
    _t_hit: float = field(default=0.0, repr=False)
    _t_miss: float = field(default=0.0, repr=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    @property
    def mean_hit_ms(self) -> float:
        return 1000 * self._t_hit / self.hits if self.hits else 0.0


class Cache:
    """On-disk per-residue embedding cache.

    Not thread-safe for concurrent writes to the same key; writes are atomic
    (temp file plus rename) so a crash cannot leave a half-written entry that a
    later run would read as valid.
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        max_bytes: int | None = None,
        dtype: str = "float16",
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.dtype = dtype
        self.stats = CacheStats()

    # ---- keys ----

    def _path(self, backbone: str, seq_hash: str) -> Path:
        safe = backbone.replace("/", "--")
        return self.root / safe / seq_hash[:2] / f"{seq_hash}.npy"

    def key(self, backbone: str, sequence: str) -> tuple[str, str]:
        return backbone, sequence_hash(sequence)

    # ---- access ----

    def get(self, backbone: str, sequence: str) -> np.ndarray | None:
        p = self._path(backbone, sequence_hash(sequence))
        t0 = time.perf_counter()
        if not p.exists():
            self.stats.misses += 1
            self.stats._t_miss += time.perf_counter() - t0
            return None
        arr = np.load(p)
        self.stats.hits += 1
        self.stats._t_hit += time.perf_counter() - t0
        return arr

    def put(self, backbone: str, sequence: str, embedding: np.ndarray) -> Path:
        if embedding.ndim != 2:
            raise ValueError(
                f"expected a per-residue (L, d) embedding, got shape {embedding.shape}. "
                "Pooling must happen at the reward head, never here — see proposal.tex §4.3."
            )
        p = self._path(backbone, sequence_hash(sequence))
        p.parent.mkdir(parents=True, exist_ok=True)
        # np.save appends '.npy' unless the name already ends in it, so write
        # through an open handle: passing a '.tmp' path would silently produce
        # '<name>.tmp.npy' and the rename below would find nothing.
        tmp = p.with_name(p.name + ".tmp")
        with tmp.open("wb") as fh:
            np.save(fh, embedding.astype(self.dtype, copy=False))
        tmp.replace(p)  # atomic
        self.stats.writes += 1
        if self.max_bytes is not None:
            self.evict_to_fit()
        return p

    def __contains__(self, item: tuple[str, str]) -> bool:
        backbone, sequence = item
        return self._path(backbone, sequence_hash(sequence)).exists()

    # ---- housekeeping ----

    def size_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.root.rglob("*.npy"))

    def count(self) -> int:
        return sum(1 for _ in self.root.rglob("*.npy"))

    def evict_to_fit(self) -> int:
        """Least-recently-used eviction down to `max_bytes`. Returns bytes freed."""
        if self.max_bytes is None:
            return 0
        files = sorted(self.root.rglob("*.npy"), key=lambda f: f.stat().st_atime)
        total = sum(f.stat().st_size for f in files)
        freed = 0
        for f in files:
            if total - freed <= self.max_bytes:
                break
            n = f.stat().st_size
            f.unlink(missing_ok=True)
            freed += n
            self.stats.evictions += 1
        return freed

    def clear(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- reporting ----

    def budget_report(self, workloads: dict[str, tuple[int, int, int]]) -> str:
        """Storage estimate for planned workloads: {name: (n_seqs, length, dim)}.

        Call this *before* filling the cache. Per-residue embeddings scale with
        length, and this is where a naive run fills the disk overnight.
        """
        lines = [f"{'workload':<28} {'n':>10} {'L':>6} {'d':>6} {'size':>12}"]
        total = 0
        for name, (n, L, d) in workloads.items():
            b = estimate_bytes(n, L, d, self.dtype)
            total += b
            lines.append(f"{name:<28} {n:>10,} {L:>6} {d:>6} {format_bytes(b):>12}")
        lines.append(f"{'TOTAL':<28} {'':>10} {'':>6} {'':>6} {format_bytes(total):>12}")
        if self.max_bytes:
            lines.append(f"{'budget':<28} {'':>10} {'':>6} {'':>6} {format_bytes(self.max_bytes):>12}")
            if total > self.max_bytes:
                lines.append(f"  OVER BUDGET by {format_bytes(total - self.max_bytes)} — eviction will thrash")
        return "\n".join(lines)

    def manifest(self) -> dict:
        return {
            "root": str(self.root),
            "dtype": self.dtype,
            "entries": self.count(),
            "bytes": self.size_bytes(),
            "max_bytes": self.max_bytes,
            "stats": {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "writes": self.stats.writes,
                "evictions": self.stats.evictions,
                "hit_rate": round(self.stats.hit_rate, 4),
            },
        }

    def write_manifest(self, path: str | Path | None = None) -> Path:
        p = Path(path) if path else self.root / "manifest.json"
        p.write_text(json.dumps(self.manifest(), indent=2) + "\n")
        return p
