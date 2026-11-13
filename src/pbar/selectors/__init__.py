from .base import Selector
from .baselines import FullCE, MIPSOnly, SingleStage
from .morph import BoundaryMORPH

REGISTRY = {"mips": MIPSOnly, "single_stage": SingleStage, "full_ce": FullCE,
            "morph": BoundaryMORPH}

try:  # batch variants (Milestone 3)
    from .batch import BATCH_SELECTORS
    REGISTRY.update(BATCH_SELECTORS)
except ImportError:
    pass


def make(name: str, **kwargs) -> Selector:
    return REGISTRY[name](**kwargs)
