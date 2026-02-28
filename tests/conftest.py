"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def _warmup_torch_dynamo():
    """Pre-pay the one-time PyTorch dynamo import cost (~4s).

    The first call to torch.optim.Adam triggers a lazy import chain
    (torch._dynamo, torch._functorch, etc.). Running it once at session
    start prevents this cost from inflating any individual test's duration.
    """
    _dummy = torch.nn.Linear(1, 1)
    _opt = torch.optim.Adam(_dummy.parameters(), lr=1e-3)
    del _opt, _dummy


@pytest.fixture
def device() -> torch.device:
    """Get compute device (CPU or CUDA)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def simple_reactor_params() -> dict:
    """Simple CSTR parameters for testing."""
    return {
        "V": 100.0,  # L
        "F": 10.0,  # L/min
        "C_feed": [1.0, 0.0],  # mol/L
        "T_feed": 350.0,  # K
    }


@pytest.fixture
def simple_kinetics_params() -> dict:
    """Simple Arrhenius kinetics A -> B."""
    return {
        "k0": [1e10],  # 1/min
        "Ea": [50000.0],  # J/mol
        "stoich": np.array([[-1, 1]]),  # A -> B
    }


@pytest.fixture
def synthetic_trajectory() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic trajectory for testing Neural ODE.

    Returns:
        Tuple of (z0, t_span, z_trajectory).
    """
    batch_size = 32
    state_dim = 2
    num_times = 50

    z0 = torch.randn(batch_size, state_dim)
    t_span = torch.linspace(0, 10, num_times)

    # Simple exponential decay
    z_trajectory = z0.unsqueeze(1) * torch.exp(-0.1 * t_span).view(1, -1, 1)

    return z0, t_span, z_trajectory


# Property-based testing helpers
@pytest.fixture
def assert_non_negative():
    """Helper to assert all values are non-negative."""
    def _assert(tensor: torch.Tensor | np.ndarray) -> None:
        if isinstance(tensor, torch.Tensor):
            assert torch.all(tensor >= 0), "Found negative values"
        else:
            assert np.all(tensor >= 0), "Found negative values"
    return _assert


@pytest.fixture
def assert_mass_balance():
    """Helper to assert mass balance is conserved."""
    def _assert(
        initial: torch.Tensor | np.ndarray,
        final: torch.Tensor | np.ndarray,
        tol: float = 1e-6,
    ) -> None:
        if isinstance(initial, torch.Tensor):
            initial_mass = initial.sum(dim=-1)
            final_mass = final.sum(dim=-1)
            error = torch.abs(initial_mass - final_mass).max()
        else:
            initial_mass = initial.sum(axis=-1)
            final_mass = final.sum(axis=-1)
            error = np.abs(initial_mass - final_mass).max()

        assert error < tol, f"Mass balance violated: error={error:.2e}"
    return _assert
