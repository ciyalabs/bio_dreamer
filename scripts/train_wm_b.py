"""Steps 3.1-3.3 — train World Model B and log the collapse gate every run.

    python -I scripts/train_wm_b.py --dry-run
    python -I scripts/train_wm_b.py --assay GFP_AEQVI_Sarkisyan_2016 --epochs 30

Uses the **`planning` instance** (Step 0.3's two-instance rule). The `benchmark` assays
are the reward model's evaluation set and must not be trained on here.

The loss is

    L = w(epoch) * ||f(h_t, a) - sg[h_{t+1}]||^2      dynamics, ramped
      + CE(position) + CE(amino acid)                 inverse dynamics
      + reward MSE                                    gives the latent something to hold
      + lambda * SIGReg(h)                            the only anti-collapse term

with `sg` a stop-gradient and no EMA target, per the registered design. `w` is zero for
`--warmup` epochs and then ramps, so the inverse and reward heads shape the latent before
the dynamics loss -- whose target is produced by the adapter it is training -- can start
flattening it.

Collapse diagnostics are written for **every epoch**, not just the last, because the
question Step 3.2 asks is whether the representation degrades over training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from biodreamer import tracking
from biodreamer.reward.data import assays_for
from biodreamer.world_model import (
    ResidueWorldModelB,
    WorldModelB,
    build,
    diagnostics,
    dynamics_weight,
    residue_diagnostics,
    split,
)

OUT = Path("data/processed/wm_b.json")
# plan.yaml step 3.2. The ORIGINAL floor is kept and still reported: it is a fraction of
# d_model, it reads +0.022, and it is recorded as a FAIL. It was superseded on 2026-11-28
# because it is arithmetically unreachable -- the frozen encoder's own states have
# effective rank 6.6-21.8 of 1280 -- not because it was merely unmet.
RANK_FLOOR = 0.5                 # latent_effective_rank_ratio, superseded, still reported
PRESERVED_FLOOR_MEAN = 0.7       # rank_preserved_vs_input, mean over assays
PRESERVED_FLOOR_MIN = 0.5        # rank_preserved_vs_input, worst assay


def run_residue(tr, args, device) -> dict:
    """Step 3.1 proper: the per-residue state.

    The state table is a disk-backed memmap of 26-52 GB, so unlike `run_one` this never
    holds it in memory: each batch reads only the 2 * batch states it touches. That makes
    the batch size a memory decision about the GPU, not about the assay.
    """
    torch.manual_seed(args.seed)
    z = tr.z                                   # (n_states, L, 1280) memmap view
    length = z.shape[1]
    pos_all = torch.tensor(tr.position)
    aa_all = torch.tensor(tr.aa)
    y = torch.tensor(tr.score)
    y = (y - y.mean()) / (y.std() + 1e-8)

    itr, iva = split(len(tr), seed=args.seed)
    model = ResidueWorldModelB(
        d_in=z.shape[2], d_model=args.d_model, length=length, seed=args.seed,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    def gather(rows: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(z[rows], dtype=np.float32)).to(device)

    history = []
    for epoch in range(args.epochs):
        model.train()
        w = dynamics_weight(epoch, args.warmup, args.ramp)
        perm = itr[np.random.default_rng(args.seed + epoch).permutation(len(itr))]
        tot = {"dyn": 0.0, "inv": 0.0, "rew": 0.0, "sig": 0.0}
        nb = 0
        for s0 in range(0, len(perm), args.batch):
            b = perm[s0 : s0 + args.batch]
            if len(b) < 8:
                continue
            h_t = model.encode(gather(tr.src[b]))
            h_next = model.encode(gather(tr.dst[b]))
            pos = pos_all[b].to(device)
            aa = aa_all[b].to(device)
            pred = model.step(h_t, pos, aa)
            l_dyn = (pred - h_next.detach()).pow(2).flatten(1).sum(-1).mean()
            lp, la = model.inverse_logits(h_t, h_next)
            l_inv = ce(lp, pos) + ce(la, aa)
            l_rew = (model.predict_reward(h_next) - y[tr.dst[b]].to(device)).pow(2).mean()
            l_sig = model.sigreg(model.rows(h_next))
            loss = w * l_dyn + l_inv + l_rew + args.sigreg * l_sig
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot["dyn"] += float(l_dyn)
            tot["inv"] += float(l_inv)
            tot["rew"] += float(l_rew)
            tot["sig"] += float(l_sig)
            nb += 1
            if nb % 50 == 0:
                print(f"    ep{epoch} batch {nb}/{len(perm) // args.batch}", flush=True)

        vb = iva[: args.val_batch]
        d = residue_diagnostics(
            model, gather(tr.src[vb]), gather(tr.dst[vb]),
            pos_all[vb].to(device), aa_all[vb].to(device),
        )
        d.update({k: v / max(nb, 1) for k, v in tot.items()})
        d["epoch"] = epoch
        d["dynamics_weight"] = w
        history.append(d)
        print(
            f"  ep{epoch:>3d} w={w:.2f} dyn={d['dyn']:.4f} inv={d['inv']:.4f} "
            f"rew={d['rew']:.4f} sig={d['sig']:.4f} | "
            f"preserved={d['rank_preserved_vs_input']:.3f} "
            f"pos@1={d['inverse_position_top1']:.3f} "
            f"vs_identity={d['one_step_vs_identity']:.3f}",
            flush=True,
        )
    out = {"history": history, "final": history[-1], "n_transitions": len(tr),
           "length": length, "state": "residue"}
    if args.save_checkpoint:
        ck = Path(args.save_checkpoint) / f"wm_residue__{tr.assay}.pt"
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "config": vars(args),
                    "assay": tr.assay, "length": length}, ck)
        print(f"  checkpoint -> {ck} ({ck.stat().st_size / 1e6:.0f} MB)")
        out["checkpoint"] = str(ck)
    return out


def run_one(tr, args, device) -> dict:
    torch.manual_seed(args.seed)
    z = torch.tensor(tr.z, device=device)
    src = torch.tensor(tr.src, device=device)
    dst = torch.tensor(tr.dst, device=device)
    pos = torch.tensor(tr.position, device=device)
    aa = torch.tensor(tr.aa, device=device)
    y = torch.tensor(tr.score, device=device)
    y = (y - y.mean()) / (y.std() + 1e-8)

    itr, iva = split(len(tr), seed=args.seed)
    itr_t = torch.tensor(itr, device=device)
    iva_t = torch.tensor(iva, device=device)

    model = WorldModelB(
        d_in=z.shape[1], d_model=args.d_model,
        length=int(pos.max().item()) + 1, seed=args.seed,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    history = []
    for epoch in range(args.epochs):
        model.train()
        w = dynamics_weight(epoch, args.warmup, args.ramp)
        perm = itr_t[torch.randperm(len(itr_t), device=device)]
        tot = {"dyn": 0.0, "inv": 0.0, "rew": 0.0, "sig": 0.0}
        nb = 0
        for s0 in range(0, len(perm), args.batch):
            b = perm[s0 : s0 + args.batch]
            if len(b) < 8:
                continue
            # encode only the rows this batch touches. Encoding the whole state table
            # here costs ~50x on PHOT (86,313 states against a 512-row batch) and buys
            # nothing: no loss term reads a state outside the batch.
            h_t = model.encode(z[src[b]])
            h_next = model.encode(z[dst[b]])
            pred = model.step(h_t, pos[b], aa[b])
            l_dyn = (pred - h_next.detach()).pow(2).sum(-1).mean()
            lp, la = model.inverse_logits(h_t, h_next)
            l_inv = ce(lp, pos[b]) + ce(la, aa[b])
            l_rew = (model.predict_reward(h_next) - y[dst[b]]).pow(2).mean()
            l_sig = model.sigreg(h_next)
            loss = w * l_dyn + l_inv + l_rew + args.sigreg * l_sig
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot["dyn"] += float(l_dyn)
            tot["inv"] += float(l_inv)
            tot["rew"] += float(l_rew)
            tot["sig"] += float(l_sig)
            nb += 1

        d = diagnostics(model, z, src[iva_t], dst[iva_t], pos[iva_t], aa[iva_t])
        d.update({k: v / max(nb, 1) for k, v in tot.items()})
        d["epoch"] = epoch
        d["dynamics_weight"] = w
        history.append(d)
        print(
            f"  ep{epoch:>3d} w={w:.2f} dyn={d['dyn']:.4f} inv={d['inv']:.4f} "
            f"rew={d['rew']:.4f} sig={d['sig']:.4f} | rank_ratio={d['latent_effective_rank_ratio']:.3f} "
            f"pos@1={d['inverse_position_top1']:.3f} vs_identity={d['one_step_vs_identity']:.3f}",
            flush=True,
        )
    return {"history": history, "final": history[-1], "n_transitions": len(tr)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay", default=None, help="default: every planning assay")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--ramp", type=int, default=5)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--sigreg", type=float, default=1.0)
    ap.add_argument("--max-transitions", type=int, default=60000)
    ap.add_argument(
        "--min-transitions", type=int, default=1000,
        help="skip an assay with fewer usable transitions than this. A transition needs "
             "BOTH endpoints in the pooled cache, and that cache was built for the H1 and "
             "literature splits rather than whole assays, so four planning assays fall "
             "below 1,000 and would contribute noise rather than signal. Skipped assays "
             "are recorded, not dropped quietly.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--state", default="pooled", choices=["pooled", "residue"],
        help="'residue' is the state Step 3.1 registers and the one the amended Step 3.2 "
             "gate is read on; 'pooled' reproduces the superseded validation run.",
    )
    ap.add_argument("--val-batch", type=int, default=256,
                    help="per-residue only: validation transitions per diagnostic pass")
    ap.add_argument("--save-checkpoint", default=None,
                    help="per-residue only: directory for one checkpoint per assay")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    names = [args.assay] if args.assay else list(assays_for("planning"))
    device = torch.device(args.device)
    print(f"device={device}  state={args.state}  assays={len(names)}  "
          f"max_transitions={args.max_transitions}")

    results: dict = {}
    skipped: dict = {}
    for n in names:
        t0 = time.time()
        tr = build(n, max_transitions=args.max_transitions, seed=args.seed,
                   state=args.state)
        print(f"{n}: {len(tr):,} transitions over {len(tr.z):,} states, "
              f"transitions retained {tr.transition_coverage:.0%} "
              f"(variants cached {tr.coverage:.0%}) "
              f"({time.time() - t0:.0f}s to build)")
        if args.dry_run:
            continue
        if len(tr) < args.min_transitions:
            print(f"  SKIPPED: {len(tr):,} < --min-transitions {args.min_transitions:,}")
            skipped[n] = {"n_transitions": len(tr), "coverage": tr.coverage}
            continue
        results[n] = (run_residue if args.state == 'residue' else run_one)(
            tr, args, device
        )

    if args.dry_run:
        return 0

    finals = [r["final"] for r in results.values()]
    if not finals:
        # Every assay asked for fell below --min-transitions. That is a legitimate
        # outcome when the script is driven one assay at a time (phase3_pod.sh does
        # exactly that, and GCN4_YEAST has 144 transitions), so it is recorded and
        # exits clean rather than crashing the surrounding run on an empty min().
        print(f"\n  every assay was skipped below --min-transitions "
              f"{args.min_transitions:,}; no gate is read on this run")
        Path(args.out).write_text(json.dumps(
            {"summary": None, "collapse_gate_passed": None,
             "skipped_for_too_few_transitions": skipped,
             "config": vars(args), "per_assay": {}}, indent=2, default=str) + "\n")
        print(f"wrote {args.out}")
        return 0

    summary = {
        "latent_effective_rank_ratio": float(
            np.mean([f["latent_effective_rank_ratio"] for f in finals])
        ),
        "min_rank_ratio": float(min(f["latent_effective_rank_ratio"] for f in finals)),
        "input_effective_rank": float(np.mean([f["input_effective_rank"] for f in finals])),
        "rank_preserved_vs_input": float(
            np.mean([f["rank_preserved_vs_input"] for f in finals])
        ),
        "min_rank_preserved_vs_input": float(
            min(f["rank_preserved_vs_input"] for f in finals)
        ),
        "inverse_position_top1": float(np.mean([f["inverse_position_top1"] for f in finals])),
        "inverse_aa_top1": float(np.mean([f["inverse_aa_top1"] for f in finals])),
        "one_step_vs_identity": float(np.mean([f["one_step_vs_identity"] for f in finals])),
        "one_step_normalised": float(np.mean([f["one_step_normalised"] for f in finals])),
        "sigreg_stat": float(np.mean([f["sigreg_stat"] for f in finals])),
    }
    superseded_passed = summary["latent_effective_rank_ratio"] >= RANK_FLOOR
    passed = (
        summary["rank_preserved_vs_input"] >= PRESERVED_FLOOR_MEAN
        and summary["min_rank_preserved_vs_input"] >= PRESERVED_FLOOR_MIN
    )
    print(f"\n{'=' * 72}")
    for k, v in summary.items():
        print(f"  {k:34s} {v:+.4f}")

    print(f"\n  SUPERSEDED reading, reported not hidden:")
    print(f"    latent_effective_rank_ratio >= {RANK_FLOOR} (of d_model "
          f"{int(finals[0]['latent_dim'])})  -> "
          f"{'PASS' if superseded_passed else 'FAIL as registered'}")
    print(f"    the frozen encoder's OWN states have effective rank only "
          f"{summary['input_effective_rank']:.1f} of 1280, so no adapter\n"
          f"    could pass this floor on single-protein variant data. Unreachable, not "
          f"merely unmet.")

    print(f"\n  GATE (amended 2026-11-28, registered before this run):")
    print(f"    rank_preserved_vs_input mean >= {PRESERVED_FLOOR_MEAN}  "
          f"{summary['rank_preserved_vs_input']:+.4f}  "
          f"{'PASS' if summary['rank_preserved_vs_input'] >= PRESERVED_FLOOR_MEAN else 'FAIL'}")
    print(f"    rank_preserved_vs_input min  >= {PRESERVED_FLOOR_MIN}  "
          f"{summary['min_rank_preserved_vs_input']:+.4f}  "
          f"{'PASS' if summary['min_rank_preserved_vs_input'] >= PRESERVED_FLOOR_MIN else 'FAIL'}")
    print(f"    -> {'PASS -- no collapse' if passed else 'FAIL'}")
    if not passed:
        print("    The threshold was set ABOVE the 0.55 already in hand (the originally\n"
              "    proposed 0.5 would have passed), calibrated on PHOT_CHLRE preserving\n"
              "    1.10. Before applying the remedy ladder, check transition counts: on\n"
              "    the pooled run preservation tracked them 0.22/0.34/1.10 at\n"
              "    3,769/20,142/60,000. Starvation and collapse look alike in this metric\n"
              "    and the ladder only fixes one of them.")

    Path(args.out).write_text(
        json.dumps({"summary": summary, "collapse_gate_passed": passed,
                    "superseded_gate_passed": superseded_passed,
                    "thresholds": {
                        "latent_effective_rank_ratio": RANK_FLOOR,
                        "rank_preserved_vs_input_mean": PRESERVED_FLOOR_MEAN,
                        "rank_preserved_vs_input_min": PRESERVED_FLOOR_MIN,
                    },
                    "config": vars(args),
                    "skipped_for_too_few_transitions": skipped,
                    "per_assay": results}, indent=2, default=str) + "\n"
    )
    print(f"wrote {args.out}")

    with tracking.run(
        "3.2", "latent_rank",
        variant={"instance": "planning", "model": "world_model_B",
                 "state": args.state, "ema": False},
        config={k: v for k, v in vars(args).items() if k != "out"},
        seed=args.seed,
        datasets=[f"data/processed/{'residue' if args.state == 'residue' else 'pooled'}"
                  "_cache"],
        notes=("World Model B with no EMA target; SIGReg is the only anti-collapse "
               "mechanism, so the effective-rank ratio is the gate. Diagnostics are "
               "logged every epoch because the question is whether the representation "
               "degrades over training, which the loss curve cannot answer -- a "
               "collapsed model has a LOW prediction loss."),
    ) as r:
        r.log_metrics(summary)
        r.log_artifact(args.out, kind="table")
    return 0


if __name__ == "__main__":
    sys.exit(main())
