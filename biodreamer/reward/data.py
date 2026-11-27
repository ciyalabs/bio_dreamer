"""Training splits for the reward head — Step 2.1, and the protocol H1 is scored on.

Three things live here, and each one exists because getting it wrong is silent.

**The two-instance rule.** `splits/heldout.yaml` bans 8 ProteinGym assays holding 49%
of the corpus, because they contain the landscapes Phase 6 evaluates on. A `benchmark`
instance may train on them, for literature comparability; a `planning` instance may
not, and it is the only one allowed to produce a Phase 6 number. Choosing the assay
list by hand is how the two get mixed up, so it is derived from the registry here and
the firewall re-checks it.

**H1's protocol.** Train on fewer than 300 single and double mutants per assay, test on
unseen order >= 3. That is the setting Cerebra and MULTI-evolve report, which is what
makes the pre-registered `spearman_mean >= 0.55` comparable to published numbers at
all. Deviating from it quietly would make the threshold meaningless.

**Censoring.** `scripts/censoring_audit.py` measured three of the seven H1 assays
carrying a detection limit. The mask is read from its output rather than recomputed, so
the loss and the audit cannot drift apart.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PG_ZIP = Path("data/raw/proteingym/DMS_ProteinGym_substitutions.zip")
REGISTRY = Path("splits/heldout.yaml")
BUDGET = Path("data/processed/encoding_budget.json")
AUDIT = Path("data/processed/censoring_audit.json")

H1_TRAIN_CAP = 300  # registered in PREREGISTRATION.md
H1_TEST_MIN_ORDER = 3


def banned_assays() -> set[str]:
    return set(yaml.safe_load(REGISTRY.read_text())["banned_proteingym_assays"])


def assays_for(instance: str) -> list[str]:
    """Assays an instance may train on, derived from the firewall registry.

    `benchmark` may use everything, including the banned list, and may never produce a
    Phase 6 result. `planning` excludes the banned list and is the only instance
    allowed to. Anything else is rejected rather than guessed at.
    """
    if not BUDGET.exists():
        raise FileNotFoundError(f"{BUDGET} missing -- run scripts/encoding_budget.py")
    budget = json.loads(BUDGET.read_text())
    # Every assay with order>=3 variants, INCLUDING the banned ones. `h1_assays` in
    # the budget file is already firewall-filtered, so using it for both instances
    # would silently make them identical and the two-instance rule a no-op.
    pool = {a["assay"] for a in budget["assays"] if a["max_order"] >= H1_TEST_MIN_ORDER}
    if instance == "benchmark":
        return sorted(pool)
    if instance == "planning":
        return sorted(pool - banned_assays())
    raise ValueError(
        f"unknown instance {instance!r}; the two-instance rule defines exactly "
        "'benchmark' and 'planning' (RESEARCH_PLAN.md step 0.3)"
    )


def detection_limits() -> dict[str, float]:
    """Assay -> the value its detection limit sits at, for assays that have one."""
    if not AUDIT.exists():
        raise FileNotFoundError(f"{AUDIT} missing -- run scripts/censoring_audit.py")
    audit = json.loads(AUDIT.read_text())["fitness_assays"]
    return {a: r["modal_value"] for a, r in audit.items() if r.get("detection_limit")}


@dataclass
class Assay:
    """One assay's variants, with the order and censoring the losses need."""

    name: str
    sequence: np.ndarray  # mutated protein sequences
    score: np.ndarray
    order: np.ndarray
    censored: np.ndarray
    # colon-separated substitution strings ("A23V:K56E"), kept because the additive
    # null in Step 2.2 needs one coefficient per observed substitution and cannot
    # recover them from the mutated sequence without the wild type
    mutant: np.ndarray

    def __len__(self) -> int:
        return len(self.score)


def load_assay(name: str, limits: dict[str, float] | None = None) -> Assay:
    """Read one ProteinGym assay and mark its censored variants."""
    with zipfile.ZipFile(PG_ZIP) as z:
        df = pd.read_csv(
            io.BytesIO(z.read(f"DMS_ProteinGym_substitutions/{name}.csv")),
            usecols=["mutant", "mutated_sequence", "DMS_score"],
        )
    score = pd.to_numeric(df["DMS_score"], errors="coerce")
    ok = score.notna() & df["mutated_sequence"].notna()
    df, score = df[ok], score[ok].to_numpy(float)
    order = (df["mutant"].astype(str).str.count(":") + 1).to_numpy()

    limits = detection_limits() if limits is None else limits
    limit = limits.get(name)
    cens = (
        np.isclose(score, limit, rtol=0, atol=1e-6)
        if limit is not None
        else np.zeros(len(score), bool)
    )
    return Assay(
        name=name,
        sequence=df["mutated_sequence"].astype(str).str.upper().to_numpy(),
        score=score,
        order=order,
        censored=cens,
        mutant=df["mutant"].astype(str).to_numpy(),
    )


def h1_split(
    order: np.ndarray, cap: int = H1_TRAIN_CAP, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Registered H1 protocol: <=`cap` order<=2 for training, all order>=3 for test.

    The cap is sampled rather than taken head-first, because ProteinGym files are
    often ordered by position and the first 300 rows would be the N-terminus.
    """
    low = np.flatnonzero(order <= 2)
    high = np.flatnonzero(order >= H1_TEST_MIN_ORDER)
    if len(low) > cap:
        low = np.random.default_rng(seed).choice(low, size=cap, replace=False)
    return np.sort(low), high


def load_instance(
    instance: str, assays: list[str] | None = None, seed: int = 0
) -> dict[str, tuple[Assay, np.ndarray, np.ndarray]]:
    """Every assay an instance may train on, with its H1 train/test indices."""
    names = assays if assays is not None else assays_for(instance)
    if instance == "planning":
        overlap = set(names) & banned_assays()
        if overlap:
            raise RuntimeError(
                f"planning instance would train on held-out assays {sorted(overlap)}"
            )
    limits = detection_limits()
    out = {}
    for n in names:
        a = load_assay(n, limits)
        tr, te = h1_split(a.order, seed=seed)
        if len(tr) and len(te):
            out[n] = (a, tr, te)
    return out
