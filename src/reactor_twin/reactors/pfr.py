"""Plug flow reactor (PFR) implementation with Method of Lines."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from reactor_twin.reactors.base import AbstractReactor
from reactor_twin.reactors.kinetics.base import AbstractKinetics
from reactor_twin.utils.registry import REACTOR_REGISTRY

logger = logging.getLogger(__name__)


@REACTOR_REGISTRY.register("pfr")
class PlugFlowReactor(AbstractReactor):
    """Plug flow reactor (PFR) with axial dispersion.

    A tubular reactor with flow along the axial direction. Uses Method of Lines
    (MOL) to discretize the spatial dimension into N cells, converting the PDE
    to a system of ODEs.

    Governing PDE (1D advection-diffusion-reaction):
        ∂C_i/∂t = -u * ∂C_i/∂z + D * ∂²C_i/∂z² + Σ(nu_ij * r_j)

    where:
        - u: Axial velocity (m/s)
        - D: Axial dispersion coefficient (m²/s)
        - z: Axial position (m)

    Discretized to N cells using finite differences:
        dC_i,k/dt = -u * (C_i,k - C_i,k-1)/Δz + D * (C_i,k+1 - 2*C_i,k + C_i,k-1)/Δz² + r_i,k

    State variables:
        - C_i,k: Concentration of species i in cell k (mol/L)
        Total state dimension: num_species * num_cells

    Boundary conditions:
        - Inlet (z=0): C_i,0 = C_in,i (Dirichlet)
        - Outlet (z=L): ∂C_i/∂z = 0 (Neumann)

    Attributes:
        kinetics: Reaction kinetics model.
        num_cells: Number of spatial discretization cells.
        length: Reactor length (m).
        velocity: Axial velocity u (m/s).
        dispersion: Axial dispersion coefficient D (m²/s).
    """

    def __init__(
        self,
        name: str,
        num_species: int,
        params: dict[str, Any],
        kinetics: AbstractKinetics | None = None,
        num_cells: int = 50,
    ):
        """Initialize plug flow reactor.

        Args:
            name: Reactor identifier.
            num_species: Number of chemical species.
            params: Reactor parameters (L, u, D, C_in, T, etc.).
            kinetics: Reaction kinetics model.
            num_cells: Number of spatial discretization cells.
        """
        self.kinetics = kinetics
        self.num_cells = num_cells
        super().__init__(name, num_species, params)

        # Validate required parameters
        required = ["L", "u", "D", "C_in", "T"]
        for key in required:
            if key not in params:
                raise ValueError(f"Missing required parameter: {key}")

        self.length = params["L"]
        self.velocity = params["u"]
        self.dispersion = params["D"]
        self.dz = self.length / self.num_cells  # Cell size

    def _compute_state_dim(self) -> int:
        """Compute state dimension.

        Returns:
            num_species * num_cells (one concentration per species per cell).
        """
        return self.num_species * self.num_cells

    def ode_rhs(
        self,
        t: float,
        y: np.ndarray,
        u: np.ndarray | None = None,
    ) -> np.ndarray:
        """PFR ODE right-hand side (Method of Lines).

        Discretized advection-diffusion-reaction:
            dC_i,k/dt = -u * (C_i,k - C_i,k-1)/Δz + D * (C_i,k+1 - 2*C_i,k + C_i,k-1)/Δz² + r_i,k

        Args:
            t: Time (scalar).
            y: State vector [C_1,0, ..., C_1,N-1, C_2,0, ..., C_M,N-1].
                Shape: (num_species * num_cells,).
            u: Control inputs (not used in basic PFR).

        Returns:
            Time derivative dy/dt.
        """
        # Reshape state: (num_species, num_cells)
        C = y.reshape(self.num_species, self.num_cells)

        # Get inlet concentrations (boundary condition)
        C_in = np.array(self.params["C_in"])
        T = self.params["T"]

        # Initialize derivatives
        dC_dt = np.zeros_like(C)

        # Loop over spatial cells
        for k in range(self.num_cells):
            # Get concentrations at cell k
            C_k = C[:, k]

            # Compute reaction rates at cell k
            if self.kinetics is not None:
                rates_k = self.kinetics.compute_rates(C_k, T)
            else:
                rates_k = np.zeros(self.num_species)

            # Advection term: -u * dC/dz ≈ -u * (C_k - C_k-1) / dz
            if k == 0:
                # Inlet boundary: C_k-1 = C_in
                advection = -self.velocity * (C_k - C_in) / self.dz
            else:
                advection = -self.velocity * (C_k - C[:, k - 1]) / self.dz

            # Dispersion term: D * d²C/dz² ≈ D * (C_k+1 - 2*C_k + C_k-1) / dz²
            if k == 0:
                # Inlet: use C_in for left boundary
                dispersion = self.dispersion * (C[:, k + 1] - 2 * C_k + C_in) / self.dz**2
            elif k == self.num_cells - 1:
                # Outlet: Neumann BC (dC/dz = 0), ghost node C_k+1 = C_k
                dispersion = self.dispersion * (C[:, k - 1] - C_k) / self.dz**2
            else:
                dispersion = self.dispersion * (C[:, k + 1] - 2 * C_k + C[:, k - 1]) / self.dz**2

            # Total derivative: advection + dispersion + reaction
            dC_dt[:, k] = advection + dispersion + rates_k

        # Flatten back to 1D
        return dC_dt.flatten()

    def get_initial_state(self) -> np.ndarray:
        """Get initial conditions.

        Returns:
            Initial state [C_1,0, ..., C_1,N-1, ..., C_M,N-1].
                Shape: (num_species * num_cells,).
        """
        # Default: initialize all cells with inlet concentration
        # State layout: [C_0,cell_0,...,C_0,cell_N-1, C_1,cell_0,...,C_1,cell_N-1]
        C_in = np.array(self.params["C_in"])
        C0 = np.repeat(C_in, self.num_cells)  # Each species value repeated for all cells
        return C0

    def get_state_labels(self) -> list[str]:
        """Get state variable labels.

        Returns:
            List of labels like ['C_0_cell_0', 'C_0_cell_1', ...].
        """
        labels = []
        for i in range(self.num_species):
            for k in range(self.num_cells):
                labels.append(f"C_{i}_cell_{k}")
        return labels

    def get_outlet_concentrations(self, y: np.ndarray) -> np.ndarray:
        """Extract outlet concentrations (last cell).

        Args:
            y: State vector.

        Returns:
            Outlet concentrations, shape (num_species,).
        """
        C = y.reshape(self.num_species, self.num_cells)
        return C[:, -1]

    def get_axial_profile(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get axial concentration profiles.

        Args:
            y: State vector.

        Returns:
            Tuple of (z_positions, C_profiles).
                z_positions: shape (num_cells,)
                C_profiles: shape (num_species, num_cells)
        """
        C = y.reshape(self.num_species, self.num_cells)
        z_positions = np.linspace(0, self.length, self.num_cells)
        return z_positions, C

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> PlugFlowReactor:
        """Deserialize PFR from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            PlugFlowReactor instance.
        """
        return cls(
            name=config["name"],
            num_species=config["num_species"],
            params=config["params"],
            kinetics=config.get("kinetics"),
            num_cells=config.get("num_cells", 50),
        )


__all__ = ["PlugFlowReactor"]
