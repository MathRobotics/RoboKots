from dataclasses import dataclass
from typing import TYPE_CHECKING

from robokots.core.state_cache import StateCache
from robokots.inward.term import VariablePack
from robokots.inward.expr.registry import Registry

@dataclass
class BuilderContext:
    pack: VariablePack
    state_cache: StateCache
    model: object          
    registry: "Registry"   