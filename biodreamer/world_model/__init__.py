"""World Model B — deterministic latent dynamics over a frozen protein encoder (Phase 3)."""

from biodreamer.world_model.collapse import diagnostics, effective_rank
from biodreamer.world_model.model import SIGReg, WorldModelB, dynamics_weight
from biodreamer.world_model.transitions import Transitions, build, split

__all__ = [
    "SIGReg",
    "Transitions",
    "WorldModelB",
    "build",
    "diagnostics",
    "dynamics_weight",
    "effective_rank",
    "split",
]
