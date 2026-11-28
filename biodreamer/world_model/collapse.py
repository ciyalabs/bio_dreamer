"""Step 3.2 — collapse diagnostics. The gate for Phase 3.

ProteinJEPA found that latent prediction without a masked-language-modelling anchor
collapses in nearly every experiment on protein data, with gains that did not replicate
across seeds. World Model B removes the EMA target deliberately, on the LeWorldModel
design where SIGReg replaces teacher-student machinery, so the one thing that must not be
assumed is that it worked.

The registered threshold is `latent_effective_rank_ratio >= 0.5`.

**Why effective rank and not the loss.** A collapsed model has a *low* prediction loss --
predicting a constant is easy when the target is nearly constant. Training loss moving in
the right direction is consistent with total collapse, which is exactly why the gate is
written on the representation rather than on the objective.
"""

from __future__ import annotations

import numpy as np
import torch


def effective_rank(h: np.ndarray) -> float:
    """exp(entropy of the normalised eigenspectrum) -- the participation ratio of
    variance across directions.

    Hard rank counts non-zero eigenvalues and so reports `d` for a latent whose variance
    sits almost entirely in two directions. Effective rank reports ~2, which is the
    number a collapse diagnostic has to produce to be worth logging.
    """
    x = h - h.mean(0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    ev = s**2
    tot = ev.sum()
    if tot <= 0:
        return 0.0
    p = ev / tot
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


@torch.no_grad()
def diagnostics(model, z: torch.Tensor, src, dst, pos, aa, limit: int = 4096) -> dict:
    """Everything Step 3.2 asks to be logged every run."""
    model.eval()
    n = min(limit, len(src))
    h = model.encode(z)
    h_t, h_next = h[src[:n]], h[dst[:n]]
    pred = model.step(h_t, pos[:n], aa[:n])

    # SVD on the full state table is O(n d^2) and n reaches 86k here; the eigenspectrum
    # of a 4,096-row sample is the same shape, and this runs every epoch
    hn = h[:limit].detach().cpu().numpy()
    er = effective_rank(hn)
    ratio = er / hn.shape[1]
    # The INPUT's effective rank, which is the ceiling the adapter is working under.
    # Every variant of one protein is a near-identical sequence, so the frozen encoder
    # places them in a tiny subspace around the wild type: measured rank 7.5 of 1280 on
    # PHOT_CHLRE and 14.8 on CAPSD. A latent cannot span more directions than its input
    # has, so `latent_effective_rank_ratio >= 0.5` against d_model is unreachable here
    # for reasons that have nothing to do with training. The ratio against the input is
    # the diagnostic that can actually distinguish a collapsing model from a faithful
    # one, and it discriminates: PHOT preserves ~1.0 of its input rank, CAPSD ~0.3.
    zr = effective_rank(z[:limit].detach().cpu().numpy())

    lp, la = model.inverse_logits(h_t, h_next)
    pos_acc = (lp.argmax(-1) == pos[:n]).double().mean().item()
    aa_acc = (la.argmax(-1) == aa[:n]).double().mean().item()

    # a one-step predictor that ignores the action entirely is the control that matters:
    # if predicting "no change" scores as well, the dynamics head has learned nothing
    err = (pred - h_next).pow(2).sum(-1)
    null = (h_t - h_next).pow(2).sum(-1)
    scale = h_next.var(0).sum().clamp(min=1e-12)

    model.train()
    return {
        "latent_effective_rank": er,
        "latent_effective_rank_ratio": ratio,
        "input_effective_rank": zr,
        "rank_preserved_vs_input": er / max(zr, 1e-9),
        "latent_dim": float(hn.shape[1]),
        "sigreg_stat": float(model.sigreg(h[:limit]).item()),
        "inverse_position_top1": pos_acc,
        "inverse_aa_top1": aa_acc,
        "one_step_mse": float(err.mean().item()),
        "identity_baseline_mse": float(null.mean().item()),
        # < 1 means the dynamics head beats "nothing happened"; >= 1 means it does not
        "one_step_vs_identity": float((err.mean() / null.mean().clamp(min=1e-12)).item()),
        "one_step_normalised": float((err.mean() / scale).item()),
        "latent_std": float(h.std().item()),
    }


@torch.no_grad()
def residue_diagnostics(model, z_src: torch.Tensor, z_dst: torch.Tensor,
                        pos: torch.Tensor, aa: torch.Tensor,
                        row_limit: int = 20000) -> dict:
    """Step 3.2 on the PER-RESIDUE state, with the amended gate's metric.

    `diagnostics` above takes a pooled state table and index arrays into it. A per-residue
    state table is 26-52 GB per assay and lives on disk, so this takes the two batches of
    states already gathered by the caller instead.

    **Ranks are measured on ROWS**, `(B * L, d)`, for both the latent and the input. That
    is the object the amended gate names, and it is the object `state_rank_ceiling.py`
    measured the 26-70 ceiling on, so the gate, the regulariser and the published ceiling
    all read the same thing. Measuring instead on a per-state pooled vector would re-pool
    the state and reintroduce exactly the rank loss the per-residue format exists to avoid.
    """
    model.eval()
    h_t = model.encode(z_src)
    h_next = model.encode(z_dst)
    pred = model.step(h_t, pos, aa)

    rows = model.rows(h_next)
    in_rows = z_dst.reshape(-1, z_dst.shape[-1])
    if len(rows) > row_limit:
        # the SVD is O(n d^2) and n = B * L reaches 47k on CAPSD at batch 64; a fixed
        # sample keeps the eigenspectrum's shape while letting this run every epoch
        sel = torch.randperm(len(rows), device=rows.device)[:row_limit]
        rows, in_rows = rows[sel], in_rows[sel]
    hn = rows.float().cpu().numpy()
    zr = effective_rank(in_rows.float().cpu().numpy())
    er = effective_rank(hn)

    lp, la = model.inverse_logits(h_t, h_next)
    pos_acc = (lp.argmax(-1) == pos).double().mean().item()
    aa_acc = (la.argmax(-1) == aa).double().mean().item()

    err = (pred - h_next).pow(2).flatten(1).sum(-1)
    null = (h_t - h_next).pow(2).flatten(1).sum(-1)
    scale = h_next.flatten(1).var(0).sum().clamp(min=1e-12)

    model.train()
    return {
        "latent_effective_rank": er,
        "latent_effective_rank_ratio": er / hn.shape[1],
        "input_effective_rank": zr,
        # the AMENDED gate's metric (plan.yaml step 3.2, 2026-11-28)
        "rank_preserved_vs_input": er / max(zr, 1e-9),
        "latent_dim": float(hn.shape[1]),
        "sigreg_stat": float(model.sigreg(rows).item()),
        "inverse_position_top1": pos_acc,
        "inverse_aa_top1": aa_acc,
        "one_step_mse": float(err.mean().item()),
        "identity_baseline_mse": float(null.mean().item()),
        "one_step_vs_identity": float((err.mean() / null.mean().clamp(min=1e-12)).item()),
        "one_step_normalised": float((err.mean() / scale).item()),
        "latent_std": float(h_next.std().item()),
    }
