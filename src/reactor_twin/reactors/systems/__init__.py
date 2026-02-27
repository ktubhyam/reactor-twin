"""Pre-configured benchmark reaction systems."""

from __future__ import annotations

from reactor_twin.reactors.systems.bioreactor import create_bioreactor_cstr
from reactor_twin.reactors.systems.consecutive import create_consecutive_cstr
from reactor_twin.reactors.systems.exothermic_ab import create_exothermic_cstr
from reactor_twin.reactors.systems.parallel import create_parallel_cstr
from reactor_twin.reactors.systems.van_de_vusse import create_van_de_vusse_cstr

__all__ = [
    "create_bioreactor_cstr",
    "create_consecutive_cstr",
    "create_exothermic_cstr",
    "create_parallel_cstr",
    "create_van_de_vusse_cstr",
]
