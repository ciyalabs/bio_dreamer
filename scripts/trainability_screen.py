"""Step 2.1 — which assays can H1's protocol actually be run on, judged on train only.

`GCN4_YEAST` ranks backwards in every reward-head configuration, and the tempting move
is to drop it. That move would be illegitimate if it used test performance: removing the
assay you score worst on raises the mean by construction and spends the pre-registration.

So the criterion here reads **only the training split** — the <=300 order<=2 variants H1
allows. For each assay it asks whether the frozen encoder carries any signal about that
assay's measurements at all, by out-of-fold ridge regression inside the training rows.
An assay with no in-sample signal cannot be learned from, so H1's protocol is not
satisfiable on it, and that is a statement about the data and the encoder rather than
about any model's score.

Nothing here touches order>=3. Run it before reading any H1 number.

    python -I scripts/trainability_screen.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from biodreamer.protein_dreamer.cache import sequence_hash
from biodreamer.reward.data import assays_for, detection_limits, h1_split, load_assay

CACHE = Path("data/processed/pooled_cache")
BACKBONE = "westlake-repl_SaProt_650M_AF2"
OUT = Path("data/processed/trainability_screen.json")
ALPHAS = np.logspace(-1, 5, 13)
MIN_TRAIN = 150  # below this, 5-fold OOF on a 1280-d input is not informative either


def screen(assay: str, seed: int = 0) -> dict:
    a = load_assay(assay, detection_limits())
    tr, _ = h1_split(a.order, seed=seed)
    z = np.load(CACHE / f"{BACKBONE}__{assay}.npz")
    index = {k: i for i, k in enumerate(z["keys"].tolist())}
    X = z["emb"][[index[sequence_hash(s)] for s in a.sequence[tr]]].astype(np.float32)
    z.close()
    y = a.score[tr]

    oof = np.zeros(len(y))
    for i, j in KFold(5, shuffle=True, random_state=seed).split(X):
        oof[j] = RidgeCV(alphas=ALPHAS).fit(X[i], y[i]).predict(X[j])
    rho = float(spearmanr(oof, y).statistic)
    return {
        "n_train": int(len(tr)),
        "train_score_sd": float(y.std()),
        "within_train_oof_spearman": rho,
        "trainable": bool(rho > 0.0 and len(tr) >= MIN_TRAIN),
    }


def main() -> int:
    names = assays_for("benchmark")  # the superset; planning is a subset of it
    res = {n: screen(n) for n in names}
    print(f"{'assay':34s} {'n_train':>8s} {'train sd':>10s} {'OOF rho':>9s}  verdict")
    for n, r in res.items():
        print(f"{n:34s} {r['n_train']:>8,} {r['train_score_sd']:>10.3f} "
              f"{r['within_train_oof_spearman']:>+9.3f}  "
              f"{'trainable' if r['trainable'] else 'NOT TRAINABLE'}")
    bad = [n for n, r in res.items() if not r["trainable"]]
    print(f"\nnot trainable: {bad if bad else 'none'}")
    print("criterion reads the TRAINING split only; no order>=3 variant was touched.")
    OUT.write_text(json.dumps(
        {"criterion": "within-train 5-fold OOF ridge Spearman > 0 and n_train >= 150",
         "min_train": MIN_TRAIN, "assays": res, "not_trainable": bad}, indent=2) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
