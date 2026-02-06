from dataclasses import dataclass
from typing import TYPE_CHECKING

from robokots.core.state_cache import StateCache
from robokots.core.time_grid import TimeGrid
from robokots.inward.term import VariablePack
from robokots.inward.expr.registry import Registry

@dataclass
class BuilderContext:
    pack: VariablePack
    state_cache: StateCache
    time: TimeGrid
    model: object          
    registry: "Registry"   
