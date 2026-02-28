"""Base abstract class for reaction kinetics models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class AbstractKinetics(ABC):
    """Abstract base class for reaction kinetics.

    All kinetics models (Arrhenius, Langmuir-Hinshelwood, Michaelis-Menten, etc.)
    must inherit from this class and implement the rate computation.

    Attributes:
        name: Kinetics model name.
        num_reactions: Number of reactions.
        params: Dictionary of kinetic parameters.
    """

    def __init__(self, name: str, num_reactions: int, params: dict[str, Any]):
        """Initialize kinetics model.

        Args:
            name: Kinetics model identifier.
            num_reactions: Number of reactions.
            params: Dictionary of kinetic parameters (k0, Ea, etc.).
        """
        self.name = name
        self.num_reactions = num_reactions
        self.params = params
        logger.debug(
            f"Initialized {self.__class__.__name__}: name={name}, reactions={num_reactions}"
        )

    @abstractmethod
    def compute_rates(
        self,
        concentrations: npt.NDArray[Any],
        temperature: float,
    ) -> npt.NDArray[Any]:
        """Compute reaction rates.

        Args:
            concentrations: Species concentrations, shape (num_species,).
            temperature: Temperature in Kelvin.

        Returns:
            Reaction rates, shape (num_reactions,).
        """
        raise NotImplementedError("Subclasses must implement compute_rates()")

    def compute_reaction_rates(
        self,
        concentrations: npt.NDArray[Any],
        temperature: float,
    ) -> npt.NDArray[Any]:
        """Compute individual reaction rates (before stoichiometric mapping).

        Override in subclasses that distinguish between per-reaction rates
        and net species production rates. The default returns the same as
        ``compute_rates`` which is correct when ``compute_rates`` already
        returns per-reaction rates.

        Args:
            concentrations: Species concentrations, shape (num_species,).
            temperature: Temperature in Kelvin.

        Returns:
            Per-reaction rates, shape (num_reactions,).
        """
        return self.compute_rates(concentrations, temperature)

    def validate_parameters(self) -> bool:
        """Validate kinetic parameters are physically reasonable.

        Returns:
            True if parameters are valid, False otherwise.
        """
        # Default: No validation (override in subclasses)
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize kinetics configuration to dictionary.

        Returns:
            Dictionary with kinetics type and parameters.
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "num_reactions": self.num_reactions,
            "params": self.params,
        }

    @classmethod
    @abstractmethod
    def from_dict(cls, config: dict[str, Any]) -> AbstractKinetics:
        """Deserialize kinetics from configuration dictionary.

        Args:
            config: Configuration dictionary from to_dict().

        Returns:
            Kinetics instance.
        """
        raise NotImplementedError("Subclasses must implement from_dict()")

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}', reactions={self.num_reactions})"


__all__ = ["AbstractKinetics"]
