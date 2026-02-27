"""Pre-configured benchmark reaction systems."""

from __future__ import annotations

from reactor_twin.reactors.systems.exothermic_ab import create_exothermic_cstr
from reactor_twin.reactors.systems.van_de_vusse import create_van_de_vusse_cstr

__all__ = [
    "create_exothermic_cstr",
    "create_van_de_vusse_cstr",
]
