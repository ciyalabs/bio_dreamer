"""World Model B — deterministic latent dynamics over a frozen protein encoder (Phase 3)."""

from biodreamer.world_model.collapse import (
    diagnostics,
    effective_rank,
    residue_diagnostics,
)
from biodreamer.world_model.model import (
    ResidueWorldModelB,
    SIGReg,
    WorldModelB,
    dynamics_weight,
)
from biodreamer.world_model.transitions import (
    Transitions,
    build,
    split,
    transition_index,
)

__all__ = [
    "ResidueWorldModelB",
    "SIGReg",
    "Transitions",
    "WorldModelB",
    "build",
    "diagnostics",
    "dynamics_weight",
    "effective_rank",
    "residue_diagnostics",
    "split",
    "transition_index",
]
