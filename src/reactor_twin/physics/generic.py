"""GENERIC (General Equation for Non-Equilibrium Reversible-Irreversible Coupling) framework."""

from __future__ import annotations

import logging

import torch
from torch import nn

from reactor_twin.physics.constraints import AbstractConstraint
from reactor_twin.utils.registry import CONSTRAINT_REGISTRY

logger = logging.getLogger(__name__)


@CONSTRAINT_REGISTRY.register("generic")
class GENERICConstraint(AbstractConstraint):
    """Enforce GENERIC structure for thermodynamically consistent dynamics.

    GENERIC extends Port-Hamiltonian to include irreversible processes (entropy production).

    GENERIC form:
        dz/dt = L * ∇E(z) + M * ∇S(z)

    where:
    - L is anti-symmetric and positive semi-definite (reversible part, Hamiltonian)
    - M is symmetric and positive semi-definite (irreversible part, dissipative)
    - E(z) is total energy
    - S(z) is entropy
    - Degeneracy conditions: L * ∇S = 0, M * ∇E = 0 (energy conserved by reversible, entropy by dissipative)

    Hard mode: Parameterize ODE function to respect GENERIC structure.
    Soft mode: Penalize deviation from GENERIC structure and degeneracy conditions.

    Attributes:
        state_dim: Dimension of state space.
        learnable_L: If True, learn L matrix (constrained to be anti-symmetric PSD).
        learnable_M: If True, learn M matrix (constrained to be symmetric PSD).
        learnable_E: If True, learn energy function E(z) via neural network.
        learnable_S: If True, learn entropy function S(z) via neural network.
    """

    def __init__(
        self,
        name: str = "generic",
        mode: str = "hard",
        weight: float = 1.0,
        state_dim: int | None = None,
        learnable_L: bool = True,
        learnable_M: bool = True,
        learnable_E: bool = True,
        learnable_S: bool = True,
    ):
        """Initialize GENERIC constraint.

        Args:
            name: Constraint identifier.
            mode: 'hard' or 'soft'.
            weight: Weight for soft constraint penalty.
            state_dim: Dimension of state space (required for hard mode).
            learnable_L: Learn reversible matrix L.
            learnable_M: Learn irreversible matrix M.
            learnable_E: Learn energy function E(z) via neural network.
            learnable_S: Learn entropy function S(z) via neural network.
        """
        super().__init__(name, mode, weight)
        self.state_dim = state_dim
        self.learnable_L = learnable_L
        self.learnable_M = learnable_M
        self.learnable_E = learnable_E
        self.learnable_S = learnable_S

        if mode == "hard" and state_dim is None:
            raise ValueError("state_dim required for hard GENERIC constraint")

        if mode == "hard" and state_dim is not None:
            self._initialize_structure()

        logger.debug(
            f"Initialized GENERICConstraint: "
            f"state_dim={state_dim}, learnable_L={learnable_L}, learnable_M={learnable_M}"
        )

    def _initialize_structure(self) -> None:
        """Initialize GENERIC matrices and energy/entropy networks."""
        assert self.state_dim is not None

        # Reversible matrix L: anti-symmetric and PSD
        # L = A - A^T where A is learned, then project to PSD
        if self.learnable_L:
            self.L_param = nn.Parameter(torch.randn(self.state_dim, self.state_dim) * 0.1)
        else:
            # Default: zeros (no reversible dynamics)
            self.register_buffer("L_param", torch.zeros(self.state_dim, self.state_dim))

        # Irreversible matrix M: symmetric and PSD
        # M = B * B^T where B is learned
        if self.learnable_M:
            self.M_factor = nn.Parameter(torch.randn(self.state_dim, self.state_dim) * 0.1)
        else:
            # Default: small identity (Fickian diffusion)
            self.register_buffer("M_factor", torch.eye(self.state_dim) * 0.1)

        # Energy function E(z)
        if self.learnable_E:
            self.energy_net = nn.Sequential(
                nn.Linear(self.state_dim, 64),
                nn.Softplus(),
                nn.Linear(64, 64),
                nn.Softplus(),
                nn.Linear(64, 1),  # Scalar output
            )
        else:
            # Default: quadratic E(z) = 0.5 * z^T z
            self.register_buffer("_quadratic_E", torch.eye(self.state_dim))

        # Entropy function S(z) (negative for stability: S = -Σ(z_i ln z_i))
        if self.learnable_S:
            self.entropy_net = nn.Sequential(
                nn.Linear(self.state_dim, 64),
                nn.Softplus(),
                nn.Linear(64, 64),
                nn.Softplus(),
                nn.Linear(64, 1),  # Scalar output
            )
        else:
            # Default: Boltzmann entropy S(z) = -Σ(z_i ln z_i) (continuous approx)
            pass  # Computed functionally

    def get_L_matrix(self) -> torch.Tensor:
        """Get reversible matrix L (anti-symmetric).

        In GENERIC, L must be anti-symmetric (L = -L^T) so that
        energy is conserved by the reversible part: z^T L z = 0.

        Returns:
            Anti-symmetric L matrix.
        """
        A = self.L_param
        L = A - A.T  # Anti-symmetric by construction
        return L

    def get_M_matrix(self) -> torch.Tensor:
        """Get irreversible matrix M (symmetric and PSD).

        Returns:
            M = B B^T (symmetric PSD).
        """
        B = self.M_factor
        M = B @ B.T
        return M

    def compute_energy(self, z: torch.Tensor) -> torch.Tensor:
        """Compute energy E(z).

        Args:
            z: State, shape (batch, state_dim).

        Returns:
            Energy values, shape (batch,).
        """
        if self.learnable_E:
            E = self.energy_net(z).squeeze(-1)
        else:
            # Quadratic: E(z) = 0.5 * z^T z
            E = 0.5 * torch.sum(z**2, dim=-1)
        return E

    def compute_entropy(self, z: torch.Tensor) -> torch.Tensor:
        """Compute entropy S(z).

        Args:
            z: State, shape (batch, state_dim).

        Returns:
            Entropy values, shape (batch,).
        """
        if self.learnable_S:
            S = self.entropy_net(z).squeeze(-1)
        else:
            # Boltzmann-like: S(z) = -Σ(z_i ln z_i)
            z_safe = torch.clamp(torch.abs(z), min=1e-8)
            S = -torch.sum(z_safe * torch.log(z_safe), dim=-1)
        return S

    def compute_gradients(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gradients ∇E(z) and ∇S(z).

        Args:
            z: State, shape (batch, state_dim).

        Returns:
            Tuple of (grad_E, grad_S), each shape (batch, state_dim).
        """
        z_copy = z.clone().requires_grad_(True)

        # Compute energy and entropy
        E = self.compute_energy(z_copy)
        S = self.compute_entropy(z_copy)

        # Compute gradients
        grad_E = torch.autograd.grad(E.sum(), z_copy, create_graph=True)[0]
        grad_S = torch.autograd.grad(S.sum(), z_copy, create_graph=True)[0]

        return grad_E, grad_S

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Compute GENERIC dynamics: dz/dt = L * ∇E(z) + M * ∇S(z).

        Args:
            z: State, shape (batch, state_dim).

        Returns:
            GENERIC dynamics dz/dt, shape (batch, state_dim).
        """
        if self.state_dim is None:
            raise ValueError("state_dim required for GENERIC dynamics")

        # Compute gradients
        grad_E, grad_S = self.compute_gradients(z)

        # Get L and M matrices
        L = self.get_L_matrix()
        M = self.get_M_matrix()

        # Compute dynamics: dz/dt = L * ∇E + M * ∇S
        reversible_part = torch.einsum("ij,bj->bi", L, grad_E)
        irreversible_part = torch.einsum("ij,bj->bi", M, grad_S)
        dz_dt = reversible_part + irreversible_part

        return dz_dt

    def compute_violation(self, z: torch.Tensor) -> torch.Tensor:
        """Compute GENERIC structure violation (soft mode).

        Penalizes:
        1. L not being anti-symmetric and PSD
        2. M not being symmetric and PSD
        3. Degeneracy conditions: L * ∇S ≈ 0, M * ∇E ≈ 0

        Args:
            z: State trajectory, shape (batch, time, state_dim).

        Returns:
            Scalar penalty for structure violation.
        """
        violation = torch.tensor(0.0, device=z.device)

        if self.state_dim is None:
            return violation

        # Use first time point for checking structure
        z_sample = z[:, 0, :] if z.ndim == 3 else z  # (batch, state_dim)

        # Get matrices
        L = self.get_L_matrix()
        M = self.get_M_matrix()

        # 1. Check L is anti-symmetric
        anti_symmetry_error = torch.norm(L + L.T)
        violation = violation + anti_symmetry_error

        # 2. Check M is symmetric
        symmetry_error = torch.norm(M - M.T)
        violation = violation + symmetry_error

        # 3. Check degeneracy conditions
        grad_E, grad_S = self.compute_gradients(z_sample)

        # L * ∇S should be ~0
        L_grad_S = torch.einsum("ij,bj->bi", L, grad_S)
        degeneracy_1 = torch.mean(L_grad_S**2)

        # M * ∇E should be ~0
        M_grad_E = torch.einsum("ij,bj->bi", M, grad_E)
        degeneracy_2 = torch.mean(M_grad_E**2)

        violation = violation + degeneracy_1 + degeneracy_2

        return violation


__all__ = ["GENERICConstraint"]
