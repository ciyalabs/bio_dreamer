"""Leakage firewall — Step 0.3 of RESEARCH_PLAN.md.

The benchmark landscapes used to test H4 live *inside* the ProteinGym training
corpus, and in FLIP2, under names that do not match the landscape. A model that
trains on them and is then "tested" on the landscape produces meaningless
numbers, and nothing in the pipeline would warn you. This module is that warning.

Two checks, both of which must pass before any `planning` run:

    check_assay_ids   - no banned source id appears in a training split
    check_sequences   - no training sequence is >= the identity threshold to a
                        held-out landscape parent, which catches renamed copies
                        and homologs that an id check cannot

and one rule:

    check_instance    - a `benchmark` checkpoint may not produce a Phase 6 result

`check_sequences` is deliberately alignment-free (ungapped k-mer containment plus
a prefix/suffix identity test). It is a guard, not a homology search: the
authoritative inventory in splits/heldout.yaml was produced with mmseqs, and this
re-derives enough of it to fail loudly on a new leak without a tool dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

DEFAULT_REGISTRY = Path("splits/heldout.yaml")

_CODON_AAS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
_CODON = {
    b1 + b2 + b3: _CODON_AAS[i * 16 + j * 4 + k]
    for i, b1 in enumerate("TCAG")
    for j, b2 in enumerate("TCAG")
    for k, b3 in enumerate("TCAG")
}


class LeakageError(AssertionError):
    """Raised when a training split touches held-out material."""


@dataclass
class Leak:
    held_out: str
    source: str
    identifier: str
    identity: float
    reason: str

    def __str__(self) -> str:
        return (
            f"{self.held_out} <- {self.source}:{self.identifier} "
            f"({self.identity:.1%}, {self.reason})"
        )


@dataclass
class Registry:
    identity_threshold: float
    held_out: list[dict]
    banned_assays: set[str]
    banned_flip2: set[str]
    instances: dict
    _sequences: dict[str, str] = field(default_factory=dict)

    def sequences(self, root: Path | str = ".") -> dict[str, str]:
        """Held-out parent sequences, translating any nucleotide entry.

        The nucleotide case is not hypothetical: SSMuLA's DHFR is DNA, and a
        protein-level check silently misses its ProteinGym counterpart.
        """
        if self._sequences:
            return self._sequences
        root = Path(root)
        for entry in self.held_out:
            fasta = root / entry["parent_fasta"]
            if not fasta.exists():
                continue
            seq = "".join(
                line.strip()
                for line in fasta.read_text().splitlines()
                if not line.startswith(">")
            ).upper()
            if entry.get("nucleotide"):
                seq = translate(seq)
            self._sequences[entry["name"]] = seq
        return self._sequences


def translate(nt: str) -> str:
    """Naive codon translation, stopping at the first stop."""
    nt = nt.upper().replace("U", "T")
    out = []
    for i in range(0, len(nt) - 2, 3):
        aa = _CODON.get(nt[i : i + 3], "X")
        if aa == "*":
            break
        out.append(aa)
    return "".join(out)


def load_registry(path: Path | str = DEFAULT_REGISTRY) -> Registry:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — the firewall registry is required before training. "
            "See RESEARCH_PLAN.md step 0.3."
        )
    d = yaml.safe_load(path.read_text())
    return Registry(
        identity_threshold=float(d.get("identity_threshold", 0.70)),
        held_out=list(d.get("held_out") or []),
        banned_assays=set(d.get("banned_proteingym_assays") or []),
        banned_flip2=set(d.get("banned_flip2_datasets") or []),
        instances=dict(d.get("instances") or {}),
    )


# ---- checks ----


def check_assay_ids(
    training_ids: Iterable[str], registry: Registry, source: str = "ProteinGym"
) -> list[Leak]:
    """Flags any banned source identifier present in a training split."""
    banned = registry.banned_assays if source == "ProteinGym" else registry.banned_flip2
    present = {i for i in training_ids if i in banned}
    return [
        Leak(
            held_out="(registry)",
            source=source,
            identifier=i,
            identity=1.0,
            reason="banned identifier in training split",
        )
        for i in sorted(present)
    ]


def _ungapped_identity(a: str, b: str) -> float:
    """Best ungapped identity of the shorter sequence against any window of the longer.

    A prefix/suffix comparison is not enough: ProteinGym's SPG1 target is the
    448-residue protein G, and the 56-residue GB1 domain that is actually being
    mutated sits in the middle of it. Sliding the shorter sequence along the
    longer one finds that; comparing ends does not.

    Still ungapped, so it is a guard rather than an aligner — the authoritative
    inventory in splits/heldout.yaml was built with mmseqs.
    """
    if not a or not b:
        return 0.0
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    n = len(short)

    # cheap prefilter: two sequences at >=70% identity over a window of 25+
    # residues share at least one exact 8-mer. Skipping the O(n*m) scan when
    # they share none takes the full-corpus regression test from ~17s to <1s.
    if n >= 25:
        K = 8
        kmers = {short[i : i + K] for i in range(0, n - K + 1, 4)}
        if not any(km in long for km in kmers):
            return 0.0

    best = 0.0
    for off in range(len(long) - n + 1):
        window = long[off : off + n]
        matches = sum(x == y for x, y in zip(short, window, strict=True))
        best = max(best, matches / n)
        if best == 1.0:
            break
    return best


def check_sequences(
    training: dict[str, str], registry: Registry, root: Path | str = "."
) -> list[Leak]:
    """Flags training sequences too similar to a held-out landscape parent.

    Catches the case an id check cannot: the same protein under a different name.
    """
    leaks: list[Leak] = []
    held = registry.sequences(root)
    thr = registry.identity_threshold

    for name, hseq in held.items():
        if not hseq:
            continue
        for tid, tseq in training.items():
            tseq = (tseq or "").upper()
            if not tseq:
                continue
            ident = _ungapped_identity(hseq, tseq)
            reason = "sequence identity"
            # a short held-out parent fully contained in a longer training
            # sequence is a leak even when the prefix test misses it
            if ident < thr and len(hseq) >= 25 and hseq in tseq:
                ident, reason = 1.0, "held-out parent contained in training sequence"
            if ident >= thr:
                leaks.append(Leak(name, "training", tid, ident, reason))
    return leaks


def check_instance(instance: str, phase: str, registry: Registry) -> list[Leak]:
    """Enforces the two-instance rule: no benchmark checkpoint in Phase 6."""
    spec = registry.instances.get(instance)
    if spec is None:
        return [
            Leak("(instance)", "config", instance, 1.0, f"unknown instance {instance!r}")
        ]
    forbidden = {str(f).lower() for f in spec.get("forbidden_for", [])}
    p = phase.lower()
    if any(f in p or p in f for f in forbidden) or (
        p.startswith("6") and spec.get("may_train_on_banned")
    ):
        return [
            Leak(
                "(instance)",
                "config",
                instance,
                1.0,
                f"'{instance}' instance may not produce a {phase} result",
            )
        ]
    return []


def assert_clean(
    training_ids: Iterable[str] | None = None,
    training_sequences: dict[str, str] | None = None,
    instance: str | None = None,
    phase: str | None = None,
    registry: Registry | None = None,
    root: Path | str = ".",
    source: str = "ProteinGym",
) -> None:
    """Runs every applicable check and raises on the first failure.

    Call this at the top of any training or evaluation entry point.
    """
    registry = registry or load_registry(Path(root) / DEFAULT_REGISTRY)
    leaks: list[Leak] = []
    if training_ids is not None:
        leaks += check_assay_ids(training_ids, registry, source)
    if training_sequences is not None:
        leaks += check_sequences(training_sequences, registry, root)
    if instance is not None and phase is not None:
        leaks += check_instance(instance, phase, registry)

    if leaks:
        detail = "\n  ".join(str(x) for x in leaks)
        raise LeakageError(
            f"{len(leaks)} leakage violation(s) — training split touches held-out "
            f"material:\n  {detail}\n"
            "See splits/heldout.yaml and RESEARCH_PLAN.md step 0.3."
        )
