"""Combinatorial fitness landscape — the object every baseline and metric reads.

A landscape is a set of variants over a fixed set of mutable sites, each with a
measured fitness, plus a designated wild type. The combinatorially complete sets
(TrpB4, TEV, GB1, ParD2/3) all fit this shape, and so does any hypercube
extracted from HIS3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SEQ_ALIASES = ("variants", "variant", "aas", "muts", "sequences", "sequence", "seq", "genotype")
logger = logging.getLogger(__name__)

FIT_ALIASES = ("fitness", "fitness mean", "score", "y", "value", "activity")
# a column counting substitutions from the wild type; its zero row IS the wild type
DIST_ALIASES = ("hd", "mutation distance", "distance", "n_mut", "num_mutations")

# Wild types that cannot be inferred from the file alone. Each is verified, not
# assumed -- the modal rule below picks a DEAD variant on every one of them.
KNOWN_WILD_TYPES = {
    "GB1": ("VDGV", "HD==0 row of the shipped SSMuLA CSV; fitness 1.0000"),
    "TEV": ("TDHS", "Mutation Distance==0 row of the shipped SSMuLA CSV; fitness 0.4144"),
    "TrpB4": ("VFVS", "Johnston 2024; verified present in the landscape at fitness 0.4081"),
}


@dataclass
class Landscape:
    """Variants over `n_sites` mutable positions, with fitness and a wild type."""

    name: str
    variants: np.ndarray  # (n, n_sites) of single-character residues
    fitness: np.ndarray  # (n,)
    wild_type: str
    # where `wild_type` came from. "modal" means it was guessed and is probably
    # wrong: see from_csv.
    wild_type_source: str = "explicit"

    _index: dict[str, float] | None = field(default=None, repr=False, compare=False)
    _alphabet: list[str] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.variants) != len(self.fitness):
            raise ValueError("variants and fitness lengths differ")
        if self.variants.ndim != 2:
            raise ValueError(f"variants must be (n, n_sites), got {self.variants.shape}")

    # ---- shape ----

    @property
    def n(self) -> int:
        return len(self.fitness)

    @property
    def n_sites(self) -> int:
        return self.variants.shape[1]

    @property
    def alphabet(self) -> list[str]:
        """Residues observed at any site.

        Cached: `neighbours` reads this per step, and recomputing it scans every
        character of the variant matrix (636k for TrpB4) on each call.
        """
        if self._alphabet is None:
            object.__setattr__(self, "_alphabet", sorted(set(self.variants.ravel().tolist())))
        return self._alphabet

    def sequences(self) -> list[str]:
        return ["".join(r) for r in self.variants]

    # ---- structure ----

    def order(self) -> np.ndarray:
        """Hamming distance from the wild type, per variant."""
        return (self.variants != np.array(list(self.wild_type))).sum(axis=1)

    def wild_type_fitness(self) -> float:
        mask = (self.variants == np.array(list(self.wild_type))).all(axis=1)
        if mask.any():
            return float(self.fitness[mask][0])
        return float(np.median(self.fitness))

    def index(self) -> dict[str, float]:
        """Sequence -> fitness, for O(1) oracle lookups during a walk.

        Cached: greedy walks call this and `neighbours` thousands of times, and
        rebuilding it each call made a 160k-variant landscape unusably slow.
        """
        if self._index is None:
            object.__setattr__(
                self, "_index", dict(zip(self.sequences(), self.fitness.tolist(), strict=True))
            )
        return self._index

    def singles(self) -> "Landscape":
        """The order<=1 sub-landscape: what an additive null is allowed to see."""
        keep = self.order() <= 1
        return Landscape(
            f"{self.name}:singles", self.variants[keep], self.fitness[keep], self.wild_type
        )

    def subset(self, mask: np.ndarray, suffix: str = "subset") -> "Landscape":
        return Landscape(f"{self.name}:{suffix}", self.variants[mask], self.fitness[mask], self.wild_type)

    def neighbours(self, seq: str) -> list[str]:
        """Sequences one substitution away that are present in this landscape."""
        present = self.index()
        out = []
        for i in range(self.n_sites):
            for a in self.alphabet:
                if a == seq[i]:
                    continue
                cand = seq[:i] + a + seq[i + 1 :]
                if cand in present:
                    out.append(cand)
        return out

    def global_optimum(self) -> tuple[str, float]:
        i = int(np.argmax(self.fitness))
        return "".join(self.variants[i]), float(self.fitness[i])

    # ---- loading ----

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        name: str | None = None,
        wild_type: str | None = None,
    ) -> "Landscape":
        """Read an SSMuLA-style landscape CSV, de-duplicating repeated genotypes.

        `wild_type` overrides inference. Without it the wild type comes from a
        substitution-distance column if the file has one, then from
        `KNOWN_WILD_TYPES`, and only then from the modal residue per site -- which
        warns, because it is usually wrong. Check `wild_type_source` to see which.
        """
        path = Path(path)
        df = pd.read_csv(path)
        df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
        cols = {c.lower(): c for c in df.columns}
        seq_col = next((cols[a] for a in SEQ_ALIASES if a in cols), None)
        fit_col = next((cols[a] for a in FIT_ALIASES if a in cols), None)
        if seq_col is None or fit_col is None:
            raise ValueError(f"{path.name}: columns not recognised: {list(df.columns)[:8]}")

        seqs = df[seq_col].astype(str).str.strip().str.upper()
        fit = pd.to_numeric(df[fit_col], errors="coerce")
        keep = fit.notna() & seqs.str.fullmatch(r"[A-Z]+")
        seqs, fit = seqs[keep], fit[keep]
        modal = int(seqs.str.len().mode()[0])
        same = seqs.str.len() == modal
        dedup = (
            pd.DataFrame({"seq": seqs[same].to_numpy(), "fit": fit[same].to_numpy()})
            .groupby("seq", as_index=False)["fit"]
            .mean()
        )
        variants = np.array([list(s) for s in dedup["seq"]])
        label = name or path.stem

        # Resolving the wild type, in order of trustworthiness.
        #
        # The modal residue per site -- what this used to do unconditionally -- is
        # meaningless on a combinatorially complete landscape: every residue appears
        # at every site by design, so the mode is whichever one survived
        # de-duplication most often. In practice it picks a DEAD variant. On GB1 it
        # returned WLVS at fitness 0.0039 instead of VDGV at 1.0000, and on TEV FLPP
        # at -0.0162 instead of TDHS at 0.4144. Everything anchored on the wild type
        # -- singles(), fit_on_singles, greedy from the wild type, path accessibility,
        # reciprocal sign epistasis, stack-top-k -- was measured from a dead variant.
        wt, source = None, ""
        if wild_type is not None:
            wt, source = wild_type.strip().upper(), "explicit"
        else:
            dist_col = next((cols[a] for a in DIST_ALIASES if a in cols), None)
            if dist_col is not None:
                d = pd.to_numeric(df[dist_col], errors="coerce")
                zero = df.loc[d == 0, seq_col].astype(str).str.strip().str.upper()
                if len(zero):
                    wt, source = zero.iloc[0], f"{dist_col}==0"
            if wt is None and label in KNOWN_WILD_TYPES:
                wt, source = KNOWN_WILD_TYPES[label][0], KNOWN_WILD_TYPES[label][1]

        if wt is None:
            wt = "".join(pd.Series(variants[:, i]).mode()[0] for i in range(modal))
            source = "modal"
            logger.warning(
                "%s: no wild type given, no distance column, and no registry entry -- "
                "falling back to the modal residue per site (%s). On a combinatorially "
                "complete landscape this is close to arbitrary and is usually a dead "
                "variant. Every wild-type-anchored metric on this landscape is "
                "provisional; pass wild_type= to fix it.",
                label,
                wt,
            )
        elif len(wt) != modal:
            raise ValueError(
                f"{label}: wild type {wt!r} has {len(wt)} sites, landscape has {modal}"
            )

        return cls(label, variants, dedup["fit"].to_numpy(), wt, wild_type_source=source)

    @classmethod
    def synthetic_additive(
        cls, n_sites: int = 3, alphabet: str = "ACDE", seed: int = 0, noise: float = 0.0
    ) -> "Landscape":
        """A landscape that is additive by construction.

        Used to validate the additive null: on this, it must recover R^2 = 1.
        Without such a case a broken null would look like a strong result.
        """
        rng = np.random.default_rng(seed)
        effects = {(i, a): rng.normal() for i in range(n_sites) for a in alphabet}
        wt = alphabet[0] * n_sites
        for i in range(n_sites):
            effects[(i, alphabet[0])] = 0.0
        combos = np.array(
            np.meshgrid(*[list(alphabet)] * n_sites, indexing="ij")
        ).reshape(n_sites, -1).T
        y = np.array([sum(effects[(i, r)] for i, r in enumerate(row)) for row in combos])
        if noise:
            y = y + rng.normal(0, noise, size=y.shape)
        return cls(f"synthetic_additive_{n_sites}", combos, y, wt)

    @classmethod
    def synthetic_sign_epistasis(cls, seed: int = 0) -> "Landscape":
        """Two sites with reciprocal sign epistasis: a valley greedy cannot cross.

        AA = 1.0 (wild type), AB = 0.2, BA = 0.2, BB = 2.0.
        Each single step from the wild type is deleterious, the double is best.
        """
        variants = np.array([list(s) for s in ("AA", "AB", "BA", "BB")])
        fitness = np.array([1.0, 0.2, 0.2, 2.0])
        return cls("synthetic_sign_epistasis", variants, fitness, "AA")
