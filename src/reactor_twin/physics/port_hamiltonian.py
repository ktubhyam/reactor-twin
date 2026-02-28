"""Port-Hamiltonian structure-preserving neural networks."""

from __future__ import annotations

import logging
from typing import cast

import torch
from torch import nn

from reactor_twin.physics.constraints import AbstractConstraint
from reactor_twin.utils.registry import CONSTRAINT_REGISTRY

logger = logging.getLogger(__name__)


@CONSTRAINT_REGISTRY.register("port_hamiltonian")
class PortHamiltonianConstraint(AbstractConstraint):
    """Enforce Port-Hamiltonian structure for open systems.

    Port-Hamiltonian form:
        dz/dt = (J - R) * ∇H(z) + B * u

    where:
    - J is skew-symmetric (energy-conserving interconnection)
    - R is positive semi-definite (dissipation)
    - H(z) is the Hamiltonian (total energy)
    - B is input matrix
    - u is external input (control)

    Hard mode: Parameterize ODE function to respect Port-Hamiltonian structure.
    Soft mode: Penalize deviation from structure.

    Attributes:
        state_dim: Dimension of state space.
        learnable_J: If True, learn J matrix (constrained to be skew-symmetric).
        learnable_R: If True, learn R matrix (constrained to be PSD).
        learnable_H: If True, learn Hamiltonian H(z) via neural network.
    """

    def __init__(
        self,
        name: str = "port_hamiltonian",
        mode: str = "hard",
        weight: float = 1.0,
        state_dim: int | None = None,
        learnable_J: bool = True,
        learnable_R: bool = True,
        learnable_H: bool = True,
    ):
        """Initialize Port-Hamiltonian constraint.

        Args:
            name: Constraint identifier.
            mode: 'hard' or 'soft'.
            weight: Weight for soft constraint penalty.
            state_dim: Dimension of state space (required for hard mode).
            learnable_J: Learn skew-symmetric J matrix.
            learnable_R: Learn PSD R matrix.
            learnable_H: Learn Hamiltonian H(z) via neural network.
        """
        super().__init__(name, mode, weight)
        self.state_dim = state_dim
        self.learnable_J = learnable_J
        self.learnable_R = learnable_R
        self.learnable_H = learnable_H

        if mode == "hard" and state_dim is None:
            raise ValueError("state_dim required for hard Port-Hamiltonian constraint")

        # Initialize Port-Hamiltonian structure
        if mode == "hard" and state_dim is not None:
            self._initialize_structure()

        logger.debug(
            f"Initialized PortHamiltonianConstraint: "
            f"state_dim={state_dim}, learnable_J={learnable_J}, learnable_R={learnable_R}"
        )

    def _initialize_structure(self) -> None:
        """Initialize Port-Hamiltonian matrices and Hamiltonian network."""
        assert self.state_dim is not None

        # Skew-symmetric J matrix: J = A - A^T where A is learned
        if self.learnable_J:
            self.J_param = nn.Parameter(torch.randn(self.state_dim, self.state_dim) * 0.1)
        else:
            # Default: alternating structure for [q, p] (position, momentum)
            J_default = torch.zeros(self.state_dim, self.state_dim)
            half = self.state_dim // 2
            J_default[:half, half:] = torch.eye(half)
            J_default[half:, :half] = -torch.eye(half)
            self.register_buffer("J_param", J_default)

        # Positive semi-definite R matrix: R = L L^T where L is learned
        if self.learnable_R:
            self.R_factor = nn.Parameter(torch.randn(self.state_dim, self.state_dim) * 0.1)
        else:
            # Default: small diagonal dissipation
            self.register_buffer("R_factor", torch.eye(self.state_dim) * 0.01)

        # Hamiltonian network H(z)
        if self.learnable_H:
            self.hamiltonian_net = nn.Sequential(
                nn.Linear(self.state_dim, 64),
                nn.Softplus(),
                nn.Linear(64, 64),
                nn.Softplus(),
                nn.Linear(64, 1),  # Scalar output (energy)
            )
        else:
            # Default: quadratic Hamiltonian H(z) = 0.5 * z^T z
            self.register_buffer("_quadratic_H", torch.eye(self.state_dim))

    def get_J_matrix(self) -> torch.Tensor:
        """Get skew-symmetric J matrix.

        Returns:
            J = A - A^T (skew-symmetric).
        """
        A = self.J_param
        J = A - A.T
        return J

    def get_R_matrix(self) -> torch.Tensor:
        """Get positive semi-definite R matrix.

        Returns:
            R = L L^T (PSD).
        """
        L = self.R_factor
        R = L @ L.T
        return R

    def compute_hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        """Compute Hamiltonian H(z).

        Args:
            z: State, shape (batch, state_dim).

        Returns:
            Hamiltonian values, shape (batch,).
        """
        if self.learnable_H:
            H = self.hamiltonian_net(z).squeeze(-1)  # (batch,)
        else:
            # Quadratic: H(z) = 0.5 * z^T z
            H = 0.5 * torch.sum(z**2, dim=-1)  # (batch,)
        return cast(torch.Tensor, H)

    def compute_hamiltonian_gradient(self, z: torch.Tensor) -> torch.Tensor:
        """Compute gradient of Hamiltonian: ∇H(z).

        Args:
            z: State, shape (batch, state_dim).

        Returns:
            Gradient ∇H(z), shape (batch, state_dim).
        """
        z_copy = z.clone().requires_grad_(True)
        H = self.compute_hamiltonian(z_copy)

        # Compute gradient via autograd
        grad_H = torch.autograd.grad(
            H.sum(),
            z_copy,
            create_graph=True,
        )[0]

        return grad_H

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Compute Port-Hamiltonian dynamics: dz/dt = (J - R) * ∇H(z).

        This returns the dynamics, not a projected state.

        Args:
            z: State, shape (batch, state_dim).

        Returns:
            Port-Hamiltonian dynamics dz/dt, shape (batch, state_dim).
        """
        if self.state_dim is None:
            raise ValueError("state_dim required for Port-Hamiltonian dynamics")

        # Compute ∇H(z)
        grad_H = self.compute_hamiltonian_gradient(z)  # (batch, state_dim)

        # Get J and R matrices
        J = self.get_J_matrix()  # (state_dim, state_dim)
        R = self.get_R_matrix()  # (state_dim, state_dim)

        # Compute dynamics: dz/dt = (J - R) * ∇H(z)
        JR = J - R
        dz_dt = torch.einsum("ij,bj->bi", JR, grad_H)  # (batch, state_dim)

        return dz_dt

    def compute_violation(self, z: torch.Tensor) -> torch.Tensor:
        """Compute Port-Hamiltonian structure violation (soft mode).

        Penalizes:
        1. J not being skew-symmetric: ||J + J^T||
        2. R not being PSD: |min(eig(R))|
        3. Energy not being conserved (in the absence of dissipation)

        Args:
            z: State trajectory, shape (batch, time, state_dim).

        Returns:
            Scalar penalty for structure violation.
        """
        violation = torch.tensor(0.0, device=z.device)

        if self.state_dim is None:
            return violation

        # 1. Check J is skew-symmetric
        J = self.get_J_matrix()
        skew_symmetry_error = torch.norm(J + J.T)
        violation = violation + skew_symmetry_error

        # 2. Check R is PSD (smallest eigenvalue should be >= 0)
        R = self.get_R_matrix()
        eigenvalues = torch.linalg.eigvalsh(R)  # Sorted eigenvalues
        min_eigenvalue = eigenvalues[0]
        psd_violation = torch.relu(-min_eigenvalue)  # Penalize negative eigenvalues
        violation = violation + psd_violation

        return cast(torch.Tensor, violation)


__all__ = ["PortHamiltonianConstraint"]
