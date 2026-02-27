"""Plugin registry system for extensible components.

Inspired by DeepXDE's loosely-coupled design. Allows users to register
custom reactors, kinetics, constraints, and Neural DE variants without
modifying library source code.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Registry:
    """Registry for plugin components.

    Example:
        >>> REACTOR_REGISTRY = Registry("reactors")
        >>> @REACTOR_REGISTRY.register("my_reactor")
        ... class MyReactor(AbstractReactor):
        ...     pass
        >>> reactor_cls = REACTOR_REGISTRY.get("my_reactor")
    """

    def __init__(self, name: str):
        """Initialize registry.

        Args:
            name: Registry name for error messages.
        """
        self.name = name
        self._registry: dict[str, type[Any]] = {}
        logger.info(f"Initialized {name} registry")

    def register(self, key: str) -> Callable[[type[T]], type[T]]:
        """Decorator to register a class.

        Args:
            key: Unique identifier for this component.

        Returns:
            Decorator function.

        Example:
            >>> @REACTOR_REGISTRY.register("cstr")
            ... class CSTRReactor(AbstractReactor):
            ...     pass
        """

        def decorator(cls: type[T]) -> type[T]:
            if key in self._registry:
                logger.warning(f"Overwriting existing {self.name} registry entry: {key}")
            self._registry[key] = cls
            logger.debug(f"Registered {self.name}: {key} -> {cls.__name__}")
            return cls

        return decorator

    def get(self, key: str) -> type[Any]:
        """Retrieve a registered class.

        Args:
            key: Component identifier.

        Returns:
            Registered class.

        Raises:
            KeyError: If key not found in registry.
        """
        if key not in self._registry:
            available = ", ".join(self._registry.keys())
            raise KeyError(f"'{key}' not found in {self.name} registry. Available: {available}")
        return self._registry[key]

    def list_keys(self) -> list[str]:
        """List all registered keys.

        Returns:
            Sorted list of registered keys.
        """
        return sorted(self._registry.keys())

    def __contains__(self, key: str) -> bool:
        """Check if key is registered."""
        return key in self._registry

    def __repr__(self) -> str:
        """String representation."""
        keys = ", ".join(self.list_keys())
        return f"Registry('{self.name}', keys=[{keys}])"


# Global registries
REACTOR_REGISTRY = Registry("reactors")
KINETICS_REGISTRY = Registry("kinetics")
CONSTRAINT_REGISTRY = Registry("constraints")
NEURAL_DE_REGISTRY = Registry("neural_des")
SOLVER_REGISTRY = Registry("solvers")
ODE_FUNC_REGISTRY = Registry("ode_funcs")
DIGITAL_TWIN_REGISTRY = Registry("digital_twin")


__all__ = [
    "Registry",
    "REACTOR_REGISTRY",
    "KINETICS_REGISTRY",
    "CONSTRAINT_REGISTRY",
    "NEURAL_DE_REGISTRY",
    "SOLVER_REGISTRY",
    "ODE_FUNC_REGISTRY",
    "DIGITAL_TWIN_REGISTRY",
]
