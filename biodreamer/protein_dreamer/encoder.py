"""Protein encoder harness — Step 0.4 of RESEARCH_PLAN.md.

One interface over the structure-aware protein language models, returning
**per-residue** representations. Pooling happens at the reward head and nowhere
else: a single substitution perturbs a mean-pooled embedding by O(1/L), and
epistasis is not representable in a pooled vector (proposal.tex §4.3). The
`(L, d)` shape is therefore enforced rather than assumed, in `encode` and again
in `Cache.put`.

Backbones, in the order the proposal prefers them:

    SaProt   default. MIT on both HuggingFace and GitHub. Consumes Foldseek 3Di
             tokens interleaved with the sequence, so Foldseek is a build
             dependency rather than an optional tool.
    ESM-2    MIT. Sequence only; the reproducibility baseline.
    ProSST   deliberately absent. Step 0.2 established it publishes no licence.

Structure is static context: it is encoded once from the reference structure and
reused for every variant, because predicted structures are near-invariant to
point mutations (proposal.tex §2.4). Nothing here folds a mutant.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .cache import Cache

logger = logging.getLogger(__name__)

# SaProt's placeholder for "structure unknown at this position"; using it for
# every residue makes SaProt behave as a sequence-only model.
UNKNOWN_3DI = "#"

FOLDSEEK_BIN = Path("tools/foldseek/bin/foldseek")


# ---- structure tokens ----


def foldseek_3di(pdb: str | Path, binary: str | Path = FOLDSEEK_BIN) -> tuple[str, str]:
    """Return `(amino_acid_sequence, 3di_sequence)` for a structure file.

    Raises FileNotFoundError with an actionable message if Foldseek is absent —
    it is not installable via pip, see DATA_REGISTER.md.
    """
    binary = Path(binary)
    if not binary.exists():
        raise FileNotFoundError(
            f"Foldseek not found at {binary}. SaProt's structure channel needs it. "
            "Install the static build from github.com/steineggerlab/foldseek/releases "
            "into tools/, or pass binary=... explicitly."
        )
    pdb = Path(pdb)
    out = pdb.with_suffix(".3di.tsv")
    subprocess.run(
        [str(binary), "structureto3didescriptor", str(pdb), str(out)],
        check=True,
        capture_output=True,
        timeout=600,
    )
    line = out.read_text().splitlines()[0].split("\t")
    aa, tdi = line[1].strip(), line[2].strip()
    if len(aa) != len(tdi):
        raise ValueError(f"foldseek returned mismatched lengths: {len(aa)} vs {len(tdi)}")
    return aa, tdi


def interleave(sequence: str, structure: str | None = None) -> str:
    """Build SaProt's structure-aware input: aa followed by lowercase 3Di, per residue.

    With `structure=None` every position gets the unknown token, which is the
    documented way to run SaProt without a structure.
    """
    sequence = sequence.strip().upper()
    if structure is None:
        structure = UNKNOWN_3DI * len(sequence)
    structure = structure.strip()
    if len(structure) != len(sequence):
        raise ValueError(
            f"sequence ({len(sequence)}) and 3Di ({len(structure)}) lengths differ"
        )
    return "".join(a + b.lower() for a, b in zip(sequence, structure, strict=True))


# ---- backbones ----


@runtime_checkable
class Backbone(Protocol):
    name: str
    dim: int

    def encode_batch(
        self, sequences: list[str], structures: list[str | None] | None = None
    ) -> list[np.ndarray]:
        """Per-residue representations, one `(L, dim)` array per input."""
        ...


@dataclass
class _HFBackbone:
    """Shared HuggingFace masked-LM plumbing for the two supported backbones."""

    name: str
    structure_aware: bool
    cache_dir: str = "models/hf"
    device: str = "cuda"
    dtype: str = "float16"
    _model: object = None
    _tok: object = None
    dim: int = 0

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.name, cache_dir=self.cache_dir)
        torch_dtype = getattr(torch, self.dtype)
        model = AutoModelForMaskedLM.from_pretrained(
            self.name, cache_dir=self.cache_dir, dtype=torch_dtype
        ).eval()
        if self.device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
        else:
            self.device = "cpu"
            model = model.float()
        self._model = model
        self.dim = int(model.config.hidden_size)
        logger.info("loaded %s on %s (dim=%d)", self.name, self.device, self.dim)

    def encode_batch(
        self, sequences: list[str], structures: list[str | None] | None = None
    ) -> list[np.ndarray]:
        import torch

        self.load()
        structures = structures or [None] * len(sequences)
        texts = [
            interleave(s, st) if self.structure_aware else s.strip().upper()
            for s, st in zip(sequences, structures, strict=True)
        ]
        enc = self._tok(texts, return_tensors="pt", padding=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = self._model(**enc, output_hidden_states=True).hidden_states[-1]

        out: list[np.ndarray] = []
        mask = enc["attention_mask"].bool()
        for i, seq in enumerate(sequences):
            # strip BOS/EOS and any padding, leaving exactly one row per residue
            valid = hidden[i][mask[i]][1:-1]
            arr = valid.float().cpu().numpy()
            if arr.shape[0] != len(seq.strip()):
                raise ValueError(
                    f"{self.name}: got {arr.shape[0]} residue rows for a "
                    f"{len(seq.strip())}-residue sequence"
                )
            out.append(arr)
        return out


def SaProt(**kw) -> _HFBackbone:
    """SaProt 650M — the default. MIT licensed; consumes Foldseek 3Di tokens."""
    return _HFBackbone(name="westlake-repl/SaProt_650M_AF2", structure_aware=True, **kw)


def ESM2(**kw) -> _HFBackbone:
    """ESM-2 650M — the sequence-only reproducibility baseline. MIT licensed."""
    return _HFBackbone(name="facebook/esm2_t33_650M_UR50D", structure_aware=False, **kw)


BACKBONES = {"saprot": SaProt, "esm2": ESM2}


# ---- encoder ----


class ProteinEncoder:
    """Cached per-residue encoder.

    `structure` is static context supplied once per protein, not per variant.
    """

    def __init__(
        self,
        backbone: Backbone | str = "saprot",
        cache: Cache | None = None,
        **backbone_kw,
    ) -> None:
        if isinstance(backbone, str):
            if backbone not in BACKBONES:
                raise ValueError(
                    f"unknown backbone {backbone!r}; available: {', '.join(BACKBONES)}. "
                    "ProSST is deliberately absent — it publishes no licence (DATA_REGISTER.md)."
                )
            backbone = BACKBONES[backbone](**backbone_kw)
        self.backbone = backbone
        self.cache = cache if cache is not None else Cache()

    @property
    def name(self) -> str:
        return self.backbone.name

    def _cache_key(self, sequence: str, structure: str | None) -> str:
        # structure participates in the key: the same sequence encoded with and
        # without 3Di gives different embeddings and must not collide
        return sequence if structure is None else f"{sequence}|{structure}"

    def encode(
        self,
        sequences: str | list[str],
        structure: str | None = None,
        batch_size: int = 8,
        use_cache: bool = True,
    ) -> list[np.ndarray]:
        """Per-residue `(L, d)` arrays, one per sequence. Never pooled."""
        single = isinstance(sequences, str)
        seqs = [sequences] if single else list(sequences)

        results: list[np.ndarray | None] = [None] * len(seqs)
        todo: list[int] = []
        for i, s in enumerate(seqs):
            if use_cache:
                hit = self.cache.get(self.name, self._cache_key(s, structure))
                if hit is not None:
                    results[i] = hit
                    continue
            todo.append(i)

        for start in range(0, len(todo), batch_size):
            idx = todo[start : start + batch_size]
            batch = [seqs[i] for i in idx]
            embs = self.backbone.encode_batch(batch, [structure] * len(batch))
            for i, emb in zip(idx, embs, strict=True):
                if emb.ndim != 2:
                    raise ValueError(f"backbone returned shape {emb.shape}, expected (L, d)")
                results[i] = emb
                if use_cache:
                    self.cache.put(self.name, self._cache_key(seqs[i], structure), emb)

        out = [r for r in results if r is not None]
        return out[0] if single else out
