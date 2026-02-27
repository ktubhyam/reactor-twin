"""Base abstract class for all reactor types."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class AbstractReactor(ABC):
    """Abstract base class for chemical reactors.

    All reactor types (CSTR, Batch, PFR, etc.) must inherit from this class
    and implement the abstract methods for ODE right-hand side, initial conditions,
    and parameter specifications.

    Attributes:
        name: Reactor type name.
        state_dim: Number of state variables.
        num_species: Number of chemical species.
        params: Dictionary of reactor parameters.
    """

    def __init__(self, name: str, num_species: int, params: dict[str, Any]):
        """Initialize reactor.

        Args:
            name: Reactor type identifier.
            num_species: Number of chemical species.
            params: Dictionary of reactor parameters (V, F, T, etc.).
        """
        self.name = name
        self.num_species = num_species
        self.params = params
        self.state_dim = self._compute_state_dim()
        logger.debug(
            f"Initialized {self.__class__.__name__}: "
            f"name={name}, species={num_species}, state_dim={self.state_dim}"
        )

    @abstractmethod
    def _compute_state_dim(self) -> int:
        """Compute total state dimension.

        Returns:
            State dimension (e.g., num_species + 1 for CSTR with temperature).
        """
        raise NotImplementedError("Subclasses must implement _compute_state_dim()")

    @abstractmethod
    def ode_rhs(
        self,
        t: float,
        y: np.ndarray,
        u: np.ndarray | None = None,
    ) -> np.ndarray:
        """ODE right-hand side for scipy.integrate.solve_ivp.

        Args:
            t: Time (scalar).
            y: State vector, shape (state_dim,).
            u: Control inputs, shape (input_dim,). Defaults to None.

        Returns:
            Time derivative dy/dt, shape (state_dim,).
        """
        raise NotImplementedError("Subclasses must implement ode_rhs()")

    @abstractmethod
    def get_initial_state(self) -> np.ndarray:
        """Get initial conditions for the reactor.

        Returns:
            Initial state vector y0, shape (state_dim,).
        """
        raise NotImplementedError("Subclasses must implement get_initial_state()")

    @abstractmethod
    def get_state_labels(self) -> list[str]:
        """Get human-readable labels for state variables.

        Returns:
            List of state variable names (e.g., ['C_A', 'C_B', 'T']).
        """
        raise NotImplementedError("Subclasses must implement get_state_labels()")

    def get_observable_indices(self) -> list[int]:
        """Get indices of observable state variables.

        By default, all states are observable. Override for partial observability.

        Returns:
            List of observable state indices.
        """
        return list(range(self.state_dim))

    def validate_state(self, y: np.ndarray) -> bool:
        """Validate physical constraints on state.

        Args:
            y: State vector, shape (state_dim,).

        Returns:
            True if state is physically valid, False otherwise.
        """
        # Default: Check non-negativity for concentrations
        concentrations = y[: self.num_species]
        return bool(np.all(concentrations >= 0))

    def to_dict(self) -> dict[str, Any]:
        """Serialize reactor configuration to dictionary.

        Returns:
            Dictionary with reactor type, parameters, and state dimension.
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "num_species": self.num_species,
            "state_dim": self.state_dim,
            "params": self.params,
        }

    @classmethod
    @abstractmethod
    def from_dict(cls, config: dict[str, Any]) -> AbstractReactor:
        """Deserialize reactor from configuration dictionary.

        Args:
            config: Configuration dictionary from to_dict().

        Returns:
            Reactor instance.
        """
        raise NotImplementedError("Subclasses must implement from_dict()")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"species={self.num_species}, "
            f"state_dim={self.state_dim})"
        )


__all__ = ["AbstractReactor"]
