"""HypercubeME output -> Landscape, so path analysis can go past four sites.

Every combinatorially complete landscape the project has from SSMuLA tops out at
four mutable sites, which caps every path-level question at horizon 4. The HIS3
extraction (Step 0.2, extended 2026-11-25) yields 13.5 million complete
sub-landscapes up to dimension 9 from one mutagenesis experiment, and those are
the only >=5-step ground truth available.

HypercubeME writes one line per hypercube, in three columns:

    diagonal         A143I:C141G          <from><position><to>, one per dimension
    first_genotype   140F:141C:142F:143A  <position><residue>, the whole corner
    last_genotype    140F:141G:142F:143I  the opposite corner

`Z` as the <to> residue means the mutation is absent, i.e. that position is wild
type in that corner. Positions outside the diagonal are shared by both corners
and form the fixed background. The 2^d corners are every combination of each
dimension's two states, and HypercubeME guarantees all of them were measured --
that is what makes the sub-landscape complete.

A corner with a position at wild type has no residue listed, so the single-letter
site encoding uses `WILD_TYPE` for it. The hypercube's local wild type is the
corner closest to the HIS3 wild type (the one with the most positions reverted),
because that is where a mutational walk would actually start.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .landscape import Landscape

WILD_TYPE = "-"  # a position reverted to wild type; never a real residue
WILD_TYPE_MARKER = "0Z"  # how HypercubeME spells "no mutations" in its output
_DIAGONAL = re.compile(r"^([A-Z])(\d+)([A-Z])$")
_SUBST = re.compile(r"^(\d+)([A-Z])$")

Genotype = frozenset  # frozenset[tuple[int, str]]


def load_genotype_table(path: str | Path) -> dict[Genotype, float]:
    """Read a HypercubeME input file into {frozenset of (position, residue): fitness}.

    This is the same file the extraction ran on, so every hypercube corner it
    emitted is a key here. The wild type is the empty mutation list.
    """
    table: dict[Genotype, float] = {}
    with Path(path).open() as fh:
        header = next(fh, "")
        if "fitness" not in header.lower():
            raise ValueError(f"{path}: expected a 'mut_list\\tfitness' header, got {header!r}")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            muts, fitness = parts[0], parts[1]
            try:
                table[parse_genotype(muts)] = float(fitness)
            except ValueError:
                continue
    return table


def parse_genotype(spec: str) -> Genotype:
    """`140F:141G` -> {(140, 'F'), (141, 'G')}; the wild type is the empty set.

    HypercubeME writes the wild type two ways: the input file leaves the mutation
    list empty, while the output corners spell it `0Z`. Read naively, `0Z` becomes
    a mutation at position 0 and the corner stops matching the genotype table —
    which silently discarded every hypercube anchored at the true wild type, 2.7%
    of the dimension-5 set and the most interesting ones in it.
    """
    spec = spec.strip()
    if not spec or spec == WILD_TYPE_MARKER:
        return frozenset()
    out = []
    for tok in spec.split(":"):
        tok = tok.strip()
        if tok == WILD_TYPE_MARKER:
            continue  # carries no substitution
        m = _SUBST.match(tok)
        if not m:
            raise ValueError(f"not a <position><residue> token: {tok!r}")
        out.append((int(m.group(1)), m.group(2)))
    return frozenset(out)


@dataclass(frozen=True)
class Hypercube:
    """One extracted hypercube: d positions, two states each, a fixed background."""

    positions: tuple[int, ...]
    state_a: tuple[str, ...]  # residue in first_genotype, or WILD_TYPE
    state_b: tuple[str, ...]  # residue in last_genotype, or WILD_TYPE
    background: Genotype

    @property
    def dimension(self) -> int:
        return len(self.positions)

    def corners(self) -> Iterator[tuple[str, Genotype]]:
        """Every one of the 2^d corners, as (site encoding, genotype key)."""
        for choice in itertools.product(*zip(self.state_a, self.state_b)):
            muts = {
                (p, r) for p, r in zip(self.positions, choice) if r != WILD_TYPE
            }
            yield "".join(choice), self.background | muts


def parse_hypercube(diagonal: str, first: str, last: str) -> Hypercube:
    """Build a Hypercube from one output line, checking the two corners agree.

    The diagonal already encodes both states, so it is cross-checked against the
    corner genotypes rather than trusted: a silent parse error here would produce
    landscapes whose fitnesses belong to other genotypes.
    """
    dims = []
    for tok in diagonal.strip().split(":"):
        m = _DIAGONAL.match(tok.strip())
        if not m:
            raise ValueError(f"not a <from><position><to> token: {tok!r}")
        src, pos, dst = m.group(1), int(m.group(2)), m.group(3)
        dims.append((pos, src, dst))
    dims.sort()

    g_first, g_last = dict(parse_genotype(first)), dict(parse_genotype(last))
    positions = tuple(p for p, _, _ in dims)
    state_a = tuple(g_first.get(p, WILD_TYPE) for p in positions)
    state_b = tuple(g_last.get(p, WILD_TYPE) for p in positions)

    for (pos, src, dst), a, b in zip(dims, state_a, state_b):
        want_a = WILD_TYPE if src == "Z" else src
        want_b = WILD_TYPE if dst == "Z" else dst
        if (a, b) != (want_a, want_b):
            raise ValueError(
                f"diagonal and corners disagree at position {pos}: "
                f"diagonal says {want_a}->{want_b}, corners say {a}->{b}"
            )
        if a == b:
            raise ValueError(f"position {pos} is not a dimension: both corners are {a}")

    var = set(positions)
    bg_first = frozenset((p, r) for p, r in g_first.items() if p not in var)
    bg_last = frozenset((p, r) for p, r in g_last.items() if p not in var)
    if bg_first != bg_last:
        raise ValueError(
            f"corners differ outside the diagonal: {sorted(bg_first ^ bg_last)}"
        )

    return Hypercube(positions, state_a, state_b, bg_first)


def to_landscape(
    hc: Hypercube, table: dict[Genotype, float], name: str = ""
) -> Landscape | None:
    """Assemble the 2^d corners into a Landscape, or None if any is unmeasured.

    A missing corner means the hypercube is not complete in this genotype table,
    which should not happen for output produced from the same file. Returning None
    rather than raising keeps a single bad line from aborting a whole sample, and
    the caller counts them.
    """
    seqs, vals = [], []
    for encoding, key in hc.corners():
        if key not in table:
            return None
        seqs.append(encoding)
        vals.append(table[key])

    variants = np.array([list(s) for s in seqs])
    # The local wild type is where a real mutational walk would start: the corner
    # with the most positions reverted. Where neither state is wild type the two
    # are both mutations and the count ties, so the tie goes to `first_genotype`
    # — a documented rule rather than whichever corner iteration reached first.
    wt = "".join(
        WILD_TYPE if WILD_TYPE in (a, b) else a
        for a, b in zip(hc.state_a, hc.state_b)
    )
    label = name or "hc" + "_".join(f"{p}{a}{b}" for p, a, b in zip(hc.positions, hc.state_a, hc.state_b))
    return Landscape(label, variants, np.array(vals, dtype=float), wt)


def iter_lines(path: str | Path) -> Iterator[tuple[str, str, str]]:
    """Stream (diagonal, first, last) triples from a hypercubes_N.txt file."""
    with Path(path).open() as fh:
        header = next(fh, "")
        if not header.startswith("diagonal"):
            raise ValueError(f"{path}: expected a 'diagonal ...' header, got {header!r}")
        for line in fh:
            parts = line.split()
            if len(parts) == 3:
                yield parts[0], parts[1], parts[2]
            elif len(parts) == 2:
                # a corner with no mutations at all prints as an empty field
                yield parts[0], parts[1], ""


def sample(
    path: str | Path,
    table: dict[Genotype, float],
    n: int,
    seed: int = 0,
) -> tuple[list[Landscape], dict[str, int]]:
    """Reservoir-sample `n` hypercubes from one dimension's file.

    The dimension-4 file is 263 MB, so lines are streamed and sampled rather than
    read into memory. Returns the landscapes and a count of what was rejected, so
    a parse problem shows up as a number instead of silently shrinking the sample.
    """
    rng = np.random.default_rng(seed)
    reservoir: list[tuple[str, str, str]] = []
    seen = 0
    for triple in iter_lines(path):
        seen += 1
        if len(reservoir) < n:
            reservoir.append(triple)
        else:
            j = int(rng.integers(seen))
            if j < n:
                reservoir[j] = triple

    out: list[Landscape] = []
    stats = {"seen": seen, "sampled": len(reservoir), "unparsed": 0, "incomplete": 0}
    for diagonal, first, last in reservoir:
        try:
            hc = parse_hypercube(diagonal, first, last)
        except ValueError:
            stats["unparsed"] += 1
            continue
        ls = to_landscape(hc, table)
        if ls is None:
            stats["incomplete"] += 1
            continue
        out.append(ls)
    stats["usable"] = len(out)
    return out, stats
