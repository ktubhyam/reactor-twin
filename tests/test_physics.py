"""Comprehensive tests for all physics constraints in reactor_twin.physics."""

from __future__ import annotations

import pytest
import torch

from reactor_twin.physics import (
    AbstractConstraint,
    ConstraintPipeline,
    EnergyBalanceConstraint,
    GENERICConstraint,
    MassBalanceConstraint,
    PortHamiltonianConstraint,
    PositivityConstraint,
    StoichiometricConstraint,
    ThermodynamicConstraint,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

torch.manual_seed(0)

BATCH_SIZE = 4
STATE_DIM = 3
TIME_STEPS = 10


# ---------------------------------------------------------------------------
# Helper to create tensors reproducibly
# ---------------------------------------------------------------------------

def _seeded_randn(*shape: int) -> torch.Tensor:
    """Return a randn tensor after resetting seed for reproducibility."""
    torch.manual_seed(0)
    return torch.randn(*shape)


# ===========================================================================
# AbstractConstraint base-class tests
# ===========================================================================


class TestAbstractConstraint:
    """Tests for the AbstractConstraint base class."""

    def test_invalid_mode_raises_value_error(self):
        """Providing a mode other than 'hard' or 'soft' should raise."""
        with pytest.raises(ValueError, match="mode must be 'hard' or 'soft'"):
            PositivityConstraint(name="bad", mode="invalid")

    def test_hard_mode_stores_attributes(self):
        c = PositivityConstraint(name="pos", mode="hard", weight=2.0)
        assert c.name == "pos"
        assert c.mode == "hard"
        assert c.weight == 2.0

    def test_soft_mode_stores_attributes(self):
        c = PositivityConstraint(name="pos_soft", mode="soft", weight=5.0)
        assert c.name == "pos_soft"
        assert c.mode == "soft"
        assert c.weight == 5.0

    def test_repr_contains_class_info(self):
        c = PositivityConstraint(name="pos", mode="hard", weight=1.0)
        r = repr(c)
        assert "PositivityConstraint" in r
        assert "pos" in r
        assert "hard" in r

    def test_forward_hard_returns_projected_z_and_zero_violation(self):
        """In hard mode, forward() should return (projected_z, tensor(0.0))."""
        c = PositivityConstraint(name="pos", mode="hard")
        z = _seeded_randn(BATCH_SIZE, STATE_DIM)
        z_out, violation = c(z)
        assert z_out.shape == z.shape
        assert violation.item() == 0.0

    def test_forward_soft_returns_original_z_and_weighted_violation(self):
        """In soft mode, forward() should return (z, weight * violation)."""
        weight = 3.0
        c = PositivityConstraint(name="pos", mode="soft", weight=weight)
        z = torch.tensor([[-1.0, 2.0, -0.5]])
        z_out, violation = c(z)
        # z_out should be identical to z
        torch.testing.assert_close(z_out, z)
        # violation should be weight * compute_violation(z)
        raw_violation = c.compute_violation(z)
        torch.testing.assert_close(violation, weight * raw_violation)


# ===========================================================================
# PositivityConstraint tests
# ===========================================================================


class TestPositivityConstraint:
    """Tests for the PositivityConstraint."""

    # -- Initialization --

    def test_default_init(self):
        c = PositivityConstraint()
        assert c.name == "positivity"
        assert c.mode == "hard"
        assert c.weight == 1.0
        assert c.method == "softplus"
        assert c.indices is None
        assert c.epsilon == pytest.approx(1e-8)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be"):
            PositivityConstraint(method="sigmoid")

    # -- Hard mode: softplus --

    def test_hard_softplus_all_positive(self):
        c = PositivityConstraint(mode="hard", method="softplus")
        z = _seeded_randn(BATCH_SIZE, STATE_DIM)
        z_out, _ = c(z)
        assert torch.all(z_out > 0), "softplus should produce strictly positive values"

    def test_hard_softplus_shape_2d(self):
        c = PositivityConstraint(mode="hard", method="softplus")
        z = _seeded_randn(BATCH_SIZE, STATE_DIM)
        z_out, _ = c(z)
        assert z_out.shape == (BATCH_SIZE, STATE_DIM)

    def test_hard_softplus_shape_3d(self):
        c = PositivityConstraint(mode="hard", method="softplus")
        z = _seeded_randn(BATCH_SIZE, TIME_STEPS, STATE_DIM)
        z_out, _ = c(z)
        assert z_out.shape == (BATCH_SIZE, TIME_STEPS, STATE_DIM)
        assert torch.all(z_out > 0)

    # -- Hard mode: relu --

    def test_hard_relu_all_non_negative(self):
        c = PositivityConstraint(mode="hard", method="relu")
        z = _seeded_randn(BATCH_SIZE, STATE_DIM) - 1.0
        z_out, _ = c(z)
        assert torch.all(z_out >= 0)

    def test_hard_relu_preserves_positive_values(self):
        c = PositivityConstraint(mode="hard", method="relu", epsilon=0.0)
        z = torch.tensor([[1.0, 2.0, 3.0]])
        z_out, _ = c(z)
        # relu(positive) = positive, epsilon=0 so exact
        torch.testing.assert_close(z_out, z)

    # -- Hard mode: square --

    def test_hard_square_all_positive(self):
        c = PositivityConstraint(mode="hard", method="square")
        z = _seeded_randn(BATCH_SIZE, STATE_DIM)
        z_out, _ = c(z)
        assert torch.all(z_out > 0), "square method should produce strictly positive values"

    # -- Hard mode returns zero violation --

    def test_hard_mode_violation_is_zero(self):
        for method in ("softplus", "relu", "square"):
            c = PositivityConstraint(mode="hard", method=method)
            z = _seeded_randn(BATCH_SIZE, STATE_DIM)
            _, violation = c(z)
            assert violation.item() == 0.0, f"violation should be 0 in hard mode for {method}"

    # -- Soft mode --

    def test_soft_mode_does_not_modify_input(self):
        c = PositivityConstraint(mode="soft", weight=10.0)
        z = _seeded_randn(BATCH_SIZE, STATE_DIM)
        z_out, _ = c(z)
        torch.testing.assert_close(z_out, z)

    def test_soft_mode_positive_penalty_for_negative_values(self):
        c = PositivityConstraint(mode="soft", weight=1.0)
        z = torch.tensor([[-1.0, 2.0, -0.5]])
        _, violation = c(z)
        assert violation.item() > 0

    def test_soft_mode_zero_penalty_for_all_positive(self):
        c = PositivityConstraint(mode="soft", weight=1.0)
        z = torch.tensor([[1.0, 2.0, 3.0]])
        _, violation = c(z)
        assert violation.item() == pytest.approx(0.0, abs=1e-10)

    def test_soft_penalty_weighted(self):
        weight = 7.5
        c = PositivityConstraint(mode="soft", weight=weight)
        z = torch.tensor([[-2.0, 1.0, -1.0]])
        _, violation = c(z)
        raw = c.compute_violation(z)
        torch.testing.assert_close(violation, weight * raw)

    def test_soft_penalty_is_mean_squared_negative_part(self):
        c = PositivityConstraint(mode="soft", weight=1.0)
        z = torch.tensor([[-3.0, 1.0, -1.0]])
        raw = c.compute_violation(z)
        # relu(-z) = [3, 0, 1], squared = [9, 0, 1], mean = 10/3
        expected = torch.tensor(10.0 / 3.0)
        torch.testing.assert_close(raw, expected, atol=1e-6, rtol=1e-6)

    # -- Indices --

    def test_indices_only_constrains_selected_dims(self):
        c = PositivityConstraint(mode="hard", indices=[0, 2], method="relu")
        z = torch.tensor([[-1.0, -2.0, -3.0]])
        z_out, _ = c(z)
        # Indices 0 and 2 should be non-negative
        assert z_out[0, 0].item() >= 0
        assert z_out[0, 2].item() >= 0
        # Index 1 should be unchanged (still negative)
        assert z_out[0, 1].item() == pytest.approx(-2.0)

    def test_indices_soft_only_penalizes_selected_dims(self):
        c = PositivityConstraint(mode="soft", weight=1.0, indices=[0])
        z = torch.tensor([[-1.0, -2.0, 3.0]])
        raw = c.compute_violation(z)
        # Only index 0 is checked: relu(-(-1)) = 1, squared = 1, mean = 1
        expected = torch.tensor(1.0)
        torch.testing.assert_close(raw, expected, atol=1e-6, rtol=1e-6)

    def test_indices_with_3d_input(self):
        c = PositivityConstraint(mode="hard", indices=[1], method="relu")
        z = torch.tensor([[[-1.0, -2.0, -3.0], [1.0, -0.5, 2.0]]])
        z_out, _ = c(z)
        assert z_out.shape == (1, 2, 3)
        # Only index 1 should be non-negative
        assert torch.all(z_out[..., 1] >= 0)
        # Other indices unchanged
        assert z_out[0, 0, 0].item() == pytest.approx(-1.0)
        assert z_out[0, 0, 2].item() == pytest.approx(-3.0)


# ===========================================================================
# MassBalanceConstraint tests
# ===========================================================================


class TestMassBalanceConstraint:
    """Tests for the MassBalanceConstraint."""

    # -- Initialization --

    def test_default_init(self):
        c = MassBalanceConstraint()
        assert c.name == "mass_balance"
        assert c.mode == "hard"
        assert c.check_total_mass is True
        assert c.initial_mass is None

    def test_init_with_stoich_matrix(self):
        S = torch.tensor([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
        c = MassBalanceConstraint(stoich_matrix=S)
        assert c.stoich_matrix is not None
        assert hasattr(c, "projection_matrix")

    # -- Hard mode: total mass conservation --

    def test_hard_preserves_total_mass_2d(self):
        c = MassBalanceConstraint(mode="hard", check_total_mass=True)
        z = torch.tensor([[1.0, 2.0, 3.0]])
        z_out, _ = c(z)
        original_mass = z.sum(dim=-1)
        constrained_mass = z_out.sum(dim=-1)
        torch.testing.assert_close(constrained_mass, original_mass, atol=1e-5, rtol=1e-5)

    def test_hard_preserves_initial_mass_across_calls(self):
        c = MassBalanceConstraint(mode="hard", check_total_mass=True)
        z1 = torch.tensor([[2.0, 3.0, 1.0]])  # Total = 6
        z1_out, _ = c(z1)
        initial_mass = z1.sum(dim=-1)

        z2 = torch.tensor([[3.0, 4.0, 5.0]])  # Total = 12, should be scaled to 6
        z2_out, _ = c(z2)
        constrained_mass = z2_out.sum(dim=-1)
        torch.testing.assert_close(constrained_mass, initial_mass, atol=1e-4, rtol=1e-4)

    def test_hard_3d_preserves_total_mass(self):
        c = MassBalanceConstraint(mode="hard", check_total_mass=True)
        z = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]]])
        z_out, _ = c(z)
        assert z_out.shape == z.shape
        # Each time step should have the same total as original
        original_mass = z.sum(dim=-1, keepdim=True)
        constrained_mass = z_out.sum(dim=-1, keepdim=True)
        torch.testing.assert_close(constrained_mass, original_mass, atol=1e-5, rtol=1e-5)

    def test_hard_violation_is_zero(self):
        c = MassBalanceConstraint(mode="hard", check_total_mass=True)
        z = _seeded_randn(BATCH_SIZE, STATE_DIM).abs()
        _, violation = c(z)
        assert violation.item() == 0.0

    # -- Soft mode: violation detection --

    def test_soft_violation_for_mass_change(self):
        c = MassBalanceConstraint(mode="soft", weight=1.0)
        # Trajectory where total mass changes over time
        z = torch.tensor([
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],  # mass changes at t=2
        ])
        _, violation = c(z)
        assert violation.item() > 0, "Should detect mass imbalance"

    def test_soft_zero_violation_for_conserved_mass(self):
        c = MassBalanceConstraint(mode="soft", weight=1.0)
        # Trajectory where total mass is constant
        z = torch.tensor([
            [[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [3.0, 2.0, 1.0]],
        ])
        _, violation = c(z)
        assert violation.item() == pytest.approx(0.0, abs=1e-6)

    def test_soft_mode_does_not_modify_input(self):
        c = MassBalanceConstraint(mode="soft", weight=1.0)
        z = _seeded_randn(2, TIME_STEPS, STATE_DIM)
        z_clone = z.clone()
        z_out, _ = c(z)
        torch.testing.assert_close(z_out, z_clone)

    # -- Reset --

    def test_reset_clears_initial_mass(self):
        c = MassBalanceConstraint(mode="hard")
        z = torch.tensor([[1.0, 2.0, 3.0]])
        c(z)
        assert c.initial_mass is not None
        c.reset()
        assert c.initial_mass is None

    # -- Output types and shapes --

    def test_forward_output_types(self):
        c = MassBalanceConstraint(mode="hard")
        z = _seeded_randn(BATCH_SIZE, STATE_DIM).abs()
        z_out, violation = c(z)
        assert isinstance(z_out, torch.Tensor)
        assert isinstance(violation, torch.Tensor)
        assert z_out.shape == z.shape
        assert violation.ndim == 0  # scalar


# ===========================================================================
# EnergyBalanceConstraint tests
# ===========================================================================


class TestEnergyBalanceConstraint:
    """Tests for the EnergyBalanceConstraint."""

    def test_default_init(self):
        c = EnergyBalanceConstraint()
        assert c.name == "energy_balance"
        assert c.mode == "soft"
        assert c.heat_capacity is None

    def test_project_returns_z_unchanged(self):
        """Project falls back to returning z unchanged (with warning)."""
        c = EnergyBalanceConstraint(mode="hard")
        z = _seeded_randn(BATCH_SIZE, STATE_DIM)
        z_out = c.project(z)
        torch.testing.assert_close(z_out, z)

    def test_violation_2d_returns_zero(self):
        """Energy violation is only defined for 3D (trajectory) input."""
        c = EnergyBalanceConstraint(mode="soft")
        z = _seeded_randn(BATCH_SIZE, STATE_DIM)
        violation = c.compute_violation(z)
        assert violation.item() == 0.0

    def test_violation_3d_without_cp(self):
        """Without heat capacity, should penalize temperature swings."""
        c = EnergyBalanceConstraint(mode="soft", heat_capacity=None)
        # Last dim is temperature, create trajectory with T changes
        torch.manual_seed(0)
        z = torch.randn(2, 5, 4)  # (batch=2, time=5, state_dim=4)
        violation = c.compute_violation(z)
        # Temperature changes in random data => nonzero violation
        assert violation.item() >= 0

    def test_violation_3d_with_cp(self):
        """With heat capacity, should penalize energy changes."""
        c = EnergyBalanceConstraint(mode="soft", heat_capacity=4.18)
        torch.manual_seed(0)
        z = torch.randn(2, 5, 4).abs() + 0.1
        violation = c.compute_violation(z)
        assert violation.item() >= 0

    def test_forward_soft_returns_z_and_violation(self):
        c = EnergyBalanceConstraint(mode="soft", weight=2.0)
        torch.manual_seed(0)
        z = torch.randn(2, 5, 4)
        z_out, violation = c(z)
        torch.testing.assert_close(z_out, z)
        raw = c.compute_violation(z)
        torch.testing.assert_close(violation, 2.0 * raw)

    def test_constant_trajectory_zero_violation(self):
        """A constant trajectory (no changes) should have zero violation."""
        c = EnergyBalanceConstraint(mode="soft", heat_capacity=1.0)
        # All time steps identical
        z_single = torch.tensor([[1.0, 2.0, 300.0]])  # (1, 3)
        z = z_single.unsqueeze(0).expand(2, 5, -1).clone()  # (2, 5, 3)
        violation = c.compute_violation(z)
        assert violation.item() == pytest.approx(0.0, abs=1e-6)

    def test_output_shapes(self):
        c = EnergyBalanceConstraint(mode="soft")
        z = _seeded_randn(BATCH_SIZE, TIME_STEPS, STATE_DIM + 1)
        z_out, violation = c(z)
        assert z_out.shape == z.shape
        assert violation.ndim == 0


# ===========================================================================
# ThermodynamicConstraint tests
# ===========================================================================


class TestThermodynamicConstraint:
    """Tests for the ThermodynamicConstraint."""

    def test_default_init(self):
        c = ThermodynamicConstraint()
        assert c.name == "thermodynamics"
        assert c.mode == "soft"
        assert c.check_entropy is True
        assert c.check_gibbs is True
        assert c.temperature == pytest.approx(298.15)

    def test_hard_mode_raises(self):
        """Hard mode is explicitly not supported."""
        with pytest.raises(ValueError, match="Hard thermodynamic constraints not supported"):
            ThermodynamicConstraint(mode="hard")

    def test_project_raises_not_implemented(self):
        c = ThermodynamicConstraint(mode="soft")
        z = _seeded_randn(BATCH_SIZE, STATE_DIM)
        with pytest.raises(NotImplementedError):
            c.project(z)

    def test_violation_2d_returns_zero(self):
        """2D input should return zero (only defined for trajectories)."""
        c = ThermodynamicConstraint(mode="soft")
        z = _seeded_randn(BATCH_SIZE, STATE_DIM)
        violation = c.compute_violation(z)
        assert violation.item() == 0.0

    def test_violation_3d_nonnegative(self):
        """Violation should always be non-negative."""
        c = ThermodynamicConstraint(mode="soft")
        torch.manual_seed(0)
        z = torch.randn(2, 5, 4).abs() + 0.01  # Positive concentrations + temperature
        violation = c.compute_violation(z)
        assert violation.item() >= 0

    def test_entropy_only(self):
        c = ThermodynamicConstraint(mode="soft", check_entropy=True, check_gibbs=False)
        torch.manual_seed(0)
        z = torch.randn(2, 5, 4).abs() + 0.01
        violation = c.compute_violation(z)
        assert isinstance(violation, torch.Tensor)
        assert violation.ndim == 0

    def test_gibbs_only(self):
        c = ThermodynamicConstraint(mode="soft", check_entropy=False, check_gibbs=True)
        torch.manual_seed(0)
        z = torch.randn(2, 5, 4).abs() + 0.01
        violation = c.compute_violation(z)
        assert isinstance(violation, torch.Tensor)
        assert violation.ndim == 0

    def test_neither_check_returns_zero(self):
        c = ThermodynamicConstraint(mode="soft", check_entropy=False, check_gibbs=False)
        torch.manual_seed(0)
        z = torch.randn(2, 5, 4).abs() + 0.01
        violation = c.compute_violation(z)
        assert violation.item() == pytest.approx(0.0, abs=1e-8)

    def test_forward_soft_output(self):
        weight = 3.0
        c = ThermodynamicConstraint(mode="soft", weight=weight)
        torch.manual_seed(0)
        z = torch.randn(2, 5, 4).abs() + 0.01
        z_out, violation = c(z)
        torch.testing.assert_close(z_out, z)
        raw = c.compute_violation(z)
        torch.testing.assert_close(violation, weight * raw)

    def test_equilibrium_constants_accepted(self):
        """equilibrium_constants parameter should be stored."""
        K = torch.tensor([1.0, 0.5])
        c = ThermodynamicConstraint(mode="soft", equilibrium_constants=K)
        assert c.equilibrium_constants is not None
        torch.testing.assert_close(c.equilibrium_constants, K)


# ===========================================================================
# StoichiometricConstraint tests
# ===========================================================================


class TestStoichiometricConstraint:
    """Tests for the StoichiometricConstraint."""

    @pytest.fixture
    def stoich_matrix(self):
        """A -> B reaction: stoich = [[-1, 1]]."""
        return torch.tensor([[-1.0, 1.0]])

    @pytest.fixture
    def stoich_2rxn(self):
        """A -> B, B -> C: stoich = [[-1, 1, 0], [0, -1, 1]]."""
        return torch.tensor([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])

    def test_requires_stoich_matrix(self):
        with pytest.raises(ValueError, match="stoich_matrix is required"):
            StoichiometricConstraint(stoich_matrix=None)

    def test_init(self, stoich_matrix):
        c = StoichiometricConstraint(stoich_matrix=stoich_matrix)
        assert c.num_reactions == 1
        assert c.num_species == 2

    # -- forward_stoichiometry --

    def test_forward_stoichiometry_2d(self, stoich_matrix):
        """dC/dt = nu^T * r for 2D input."""
        c = StoichiometricConstraint(stoich_matrix=stoich_matrix)
        rates = torch.tensor([[2.0]])  # (batch=1, reactions=1)
        dC_dt = c.forward_stoichiometry(rates)
        # nu^T = [[-1], [1]], r=[2] => dC/dt = [-2, 2]
        expected = torch.tensor([[-2.0, 2.0]])
        torch.testing.assert_close(dC_dt, expected)

    def test_forward_stoichiometry_3d(self, stoich_matrix):
        """dC/dt = nu^T * r for 3D (batch, time, reactions) input."""
        c = StoichiometricConstraint(stoich_matrix=stoich_matrix)
        rates = torch.tensor([[[1.0], [2.0], [3.0]]])  # (1, 3, 1)
        dC_dt = c.forward_stoichiometry(rates)
        assert dC_dt.shape == (1, 3, 2)
        expected = torch.tensor([[[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0]]])
        torch.testing.assert_close(dC_dt, expected)

    def test_forward_stoichiometry_mass_conservation(self, stoich_matrix):
        """Total dC/dt should sum to zero for mass-conserving stoichiometry."""
        c = StoichiometricConstraint(stoich_matrix=stoich_matrix)
        rates = torch.tensor([[5.0]])
        dC_dt = c.forward_stoichiometry(rates)
        total = dC_dt.sum(dim=-1)
        assert total.item() == pytest.approx(0.0, abs=1e-6)

    def test_forward_stoichiometry_2rxn(self, stoich_2rxn):
        c = StoichiometricConstraint(stoich_matrix=stoich_2rxn)
        rates = torch.tensor([[1.0, 2.0]])  # (1, 2)
        dC_dt = c.forward_stoichiometry(rates)
        # nu^T = [[-1, 0], [1, -1], [0, 1]]
        # dC/dt = nu^T @ r = [-1, 1-2, 2] = [-1, -1, 2]
        expected = torch.tensor([[-1.0, -1.0, 2.0]])
        torch.testing.assert_close(dC_dt, expected)

    def test_forward_stoichiometry_invalid_ndim(self, stoich_matrix):
        c = StoichiometricConstraint(stoich_matrix=stoich_matrix)
        with pytest.raises(ValueError, match="Unsupported rates shape"):
            c.forward_stoichiometry(torch.tensor([1.0]))  # 1D not supported

    # -- project (stoichiometric subspace projection) --

    def test_project_2d(self, stoich_matrix):
        c = StoichiometricConstraint(stoich_matrix=stoich_matrix)
        z = torch.tensor([[1.0, -1.0]])  # Already in stoich subspace
        z_proj = c.project(z)
        assert z_proj.shape == z.shape
        # [1, -1] is in the column space of [[-1], [1]], so projection should be same
        torch.testing.assert_close(z_proj, z, atol=1e-5, rtol=1e-5)

    def test_project_3d(self, stoich_matrix):
        c = StoichiometricConstraint(stoich_matrix=stoich_matrix)
        z = _seeded_randn(2, 3, 2)
        z_proj = c.project(z)
        assert z_proj.shape == (2, 3, 2)

    def test_project_invalid_ndim(self, stoich_matrix):
        c = StoichiometricConstraint(stoich_matrix=stoich_matrix)
        with pytest.raises(ValueError, match="Unsupported z shape"):
            c.project(torch.tensor([1.0, 2.0]))  # 1D not supported

    # -- compute_violation --

    def test_violation_zero_for_stoich_consistent(self, stoich_matrix):
        """A vector already in stoichiometric subspace should have zero violation."""
        c = StoichiometricConstraint(stoich_matrix=stoich_matrix, mode="soft")
        z = torch.tensor([[3.0, -3.0]])  # In span of [-1, 1]
        violation = c.compute_violation(z)
        assert violation.item() == pytest.approx(0.0, abs=1e-5)

    def test_violation_nonzero_for_inconsistent(self, stoich_matrix):
        c = StoichiometricConstraint(stoich_matrix=stoich_matrix, mode="soft")
        z = torch.tensor([[1.0, 1.0]])  # NOT in span of [-1, 1]
        violation = c.compute_violation(z)
        assert violation.item() > 0

    # -- forward() output --

    def test_forward_hard_output(self, stoich_matrix):
        c = StoichiometricConstraint(stoich_matrix=stoich_matrix, mode="hard")
        z = _seeded_randn(BATCH_SIZE, 2)
        z_out, violation = c(z)
        assert z_out.shape == z.shape
        assert violation.item() == 0.0

    def test_forward_soft_output(self, stoich_matrix):
        c = StoichiometricConstraint(stoich_matrix=stoich_matrix, mode="soft", weight=2.0)
        z = torch.tensor([[1.0, 1.0]])
        z_out, violation = c(z)
        torch.testing.assert_close(z_out, z)
        raw = c.compute_violation(z)
        torch.testing.assert_close(violation, 2.0 * raw)


# ===========================================================================
# PortHamiltonianConstraint tests
# ===========================================================================


class TestPortHamiltonianConstraint:
    """Tests for the PortHamiltonianConstraint."""

    @pytest.fixture
    def ph_hard(self):
        torch.manual_seed(0)
        return PortHamiltonianConstraint(mode="hard", state_dim=4)

    @pytest.fixture
    def ph_soft_no_dim(self):
        """Soft PH without state_dim -- structure not initialized."""
        torch.manual_seed(0)
        return PortHamiltonianConstraint(mode="soft", state_dim=None)

    @pytest.fixture
    def ph_fixed(self):
        torch.manual_seed(0)
        return PortHamiltonianConstraint(
            mode="hard", state_dim=4,
            learnable_J=False, learnable_R=False, learnable_H=False,
        )

    # -- Initialization --

    def test_hard_requires_state_dim(self):
        with pytest.raises(ValueError, match="state_dim required"):
            PortHamiltonianConstraint(mode="hard", state_dim=None)

    def test_default_init(self, ph_hard):
        assert ph_hard.state_dim == 4
        assert ph_hard.mode == "hard"

    # -- J matrix: skew-symmetry --

    def test_J_is_skew_symmetric(self, ph_hard):
        J = ph_hard.get_J_matrix()
        torch.testing.assert_close(J, -J.T, atol=1e-6, rtol=1e-6)

    def test_J_diagonal_is_zero(self, ph_hard):
        J = ph_hard.get_J_matrix()
        diag = torch.diag(J)
        torch.testing.assert_close(diag, torch.zeros(4), atol=1e-7, rtol=1e-7)

    def test_J_fixed_structure(self, ph_fixed):
        """Non-learnable J: get_J_matrix returns A - A^T where A is the default.
        Since default A has upper-right = I, lower-left = -I, applying A - A^T
        doubles it: upper-right = 2I, lower-left = -2I."""
        J = ph_fixed.get_J_matrix()
        half = 2
        # Upper right block should be 2*I (A - A^T doubles the original)
        torch.testing.assert_close(J[:half, half:], 2.0 * torch.eye(half), atol=1e-6, rtol=1e-6)
        # Lower left block should be -2*I
        torch.testing.assert_close(J[half:, :half], -2.0 * torch.eye(half), atol=1e-6, rtol=1e-6)

    # -- R matrix: positive semi-definiteness --

    def test_R_is_symmetric(self, ph_hard):
        R = ph_hard.get_R_matrix()
        torch.testing.assert_close(R, R.T, atol=1e-6, rtol=1e-6)

    def test_R_is_psd(self, ph_hard):
        R = ph_hard.get_R_matrix()
        eigenvalues = torch.linalg.eigvalsh(R)
        assert torch.all(eigenvalues >= -1e-6), f"R has negative eigenvalue: {eigenvalues.min()}"

    def test_R_shape(self, ph_hard):
        R = ph_hard.get_R_matrix()
        assert R.shape == (4, 4)

    # -- Hamiltonian --

    def test_compute_hamiltonian_output_shape(self, ph_hard):
        torch.manual_seed(0)
        z = torch.randn(BATCH_SIZE, 4)
        H = ph_hard.compute_hamiltonian(z)
        assert H.shape == (BATCH_SIZE,)

    def test_hamiltonian_quadratic_default(self, ph_fixed):
        """Non-learnable H should be H(z) = 0.5 * z^T z."""
        z = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        H = ph_fixed.compute_hamiltonian(z)
        expected = 0.5 * (1 + 4 + 9 + 16)
        assert H.item() == pytest.approx(expected, abs=1e-5)

    # -- Hamiltonian gradient --

    def test_hamiltonian_gradient_shape(self, ph_hard):
        torch.manual_seed(0)
        z = torch.randn(BATCH_SIZE, 4)
        grad_H = ph_hard.compute_hamiltonian_gradient(z)
        assert grad_H.shape == (BATCH_SIZE, 4)

    def test_hamiltonian_gradient_quadratic(self, ph_fixed):
        """For H = 0.5 * z^T z, gradient should be z."""
        z = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        grad_H = ph_fixed.compute_hamiltonian_gradient(z)
        torch.testing.assert_close(grad_H, z, atol=1e-5, rtol=1e-5)

    # -- project (dynamics) --

    def test_project_output_shape(self, ph_hard):
        torch.manual_seed(0)
        z = torch.randn(BATCH_SIZE, 4)
        dz_dt = ph_hard.project(z)
        assert dz_dt.shape == (BATCH_SIZE, 4)

    def test_project_requires_state_dim(self):
        c = PortHamiltonianConstraint(mode="soft", state_dim=None)
        with pytest.raises(ValueError, match="state_dim required"):
            c.project(torch.randn(2, 4))

    # -- compute_violation --

    def test_violation_j_skew_symmetry_always_zero(self, ph_hard):
        """By construction J = A - A^T, so J + J^T = 0; skew violation is 0."""
        torch.manual_seed(0)
        z = torch.randn(2, 5, 4)
        violation = ph_hard.compute_violation(z)
        # J is exactly skew-symmetric by construction, R is PSD by construction
        # So the violation should be very small (just numerical noise)
        J = ph_hard.get_J_matrix()
        skew_err = torch.norm(J + J.T).item()
        assert skew_err < 1e-5

    def test_violation_nonneg(self, ph_hard):
        torch.manual_seed(0)
        z = torch.randn(2, 5, 4)
        violation = ph_hard.compute_violation(z)
        assert violation.item() >= -1e-6

    def test_violation_without_state_dim_returns_zero(self):
        c = PortHamiltonianConstraint(mode="soft", state_dim=None)
        z = torch.randn(2, 5, 4)
        violation = c.compute_violation(z)
        assert violation.item() == 0.0

    # -- forward() --

    def test_forward_hard_output_types(self, ph_hard):
        torch.manual_seed(0)
        z = torch.randn(BATCH_SIZE, 4)
        z_out, violation = ph_hard(z)
        assert isinstance(z_out, torch.Tensor)
        assert isinstance(violation, torch.Tensor)
        assert z_out.shape == (BATCH_SIZE, 4)
        assert violation.item() == 0.0

    def test_forward_soft_output_types(self, ph_soft_no_dim):
        """Soft mode without initialized structure returns z unchanged and
        zero violation (since state_dim is None)."""
        torch.manual_seed(0)
        z = torch.randn(2, 5, 4)
        z_out, violation = ph_soft_no_dim(z)
        assert isinstance(z_out, torch.Tensor)
        assert isinstance(violation, torch.Tensor)
        torch.testing.assert_close(z_out, z)
        # state_dim is None so compute_violation returns 0
        assert violation.item() == pytest.approx(0.0, abs=1e-8)


# ===========================================================================
# GENERICConstraint tests
# ===========================================================================


class TestGENERICConstraint:
    """Tests for the GENERICConstraint."""

    @pytest.fixture
    def generic_hard(self):
        torch.manual_seed(0)
        return GENERICConstraint(mode="hard", state_dim=4)

    @pytest.fixture
    def generic_soft_no_dim(self):
        """Soft GENERIC without state_dim -- structure not initialized."""
        torch.manual_seed(0)
        return GENERICConstraint(mode="soft", state_dim=None)

    @pytest.fixture
    def generic_fixed(self):
        torch.manual_seed(0)
        return GENERICConstraint(
            mode="hard", state_dim=4,
            learnable_L=False, learnable_M=False,
            learnable_E=False, learnable_S=False,
        )

    # -- Initialization --

    def test_hard_requires_state_dim(self):
        with pytest.raises(ValueError, match="state_dim required"):
            GENERICConstraint(mode="hard", state_dim=None)

    def test_default_init(self, generic_hard):
        assert generic_hard.state_dim == 4
        assert generic_hard.mode == "hard"

    # -- L matrix: anti-symmetry --

    def test_L_is_anti_symmetric(self, generic_hard):
        L = generic_hard.get_L_matrix()
        torch.testing.assert_close(L, -L.T, atol=1e-6, rtol=1e-6)

    def test_L_diagonal_is_zero(self, generic_hard):
        L = generic_hard.get_L_matrix()
        diag = torch.diag(L)
        torch.testing.assert_close(diag, torch.zeros(4), atol=1e-7, rtol=1e-7)

    def test_L_shape(self, generic_hard):
        L = generic_hard.get_L_matrix()
        assert L.shape == (4, 4)

    def test_L_fixed_is_zero(self, generic_fixed):
        """Non-learnable L defaults to zero matrix."""
        L = generic_fixed.get_L_matrix()
        torch.testing.assert_close(L, torch.zeros(4, 4), atol=1e-7, rtol=1e-7)

    # -- M matrix: symmetry and PSD --

    def test_M_is_symmetric(self, generic_hard):
        M = generic_hard.get_M_matrix()
        torch.testing.assert_close(M, M.T, atol=1e-6, rtol=1e-6)

    def test_M_is_psd(self, generic_hard):
        M = generic_hard.get_M_matrix()
        eigenvalues = torch.linalg.eigvalsh(M)
        assert torch.all(eigenvalues >= -1e-6), f"M has negative eigenvalue: {eigenvalues.min()}"

    def test_M_shape(self, generic_hard):
        M = generic_hard.get_M_matrix()
        assert M.shape == (4, 4)

    # -- Energy --

    def test_compute_energy_shape(self, generic_hard):
        torch.manual_seed(0)
        z = torch.randn(BATCH_SIZE, 4)
        E = generic_hard.compute_energy(z)
        assert E.shape == (BATCH_SIZE,)

    def test_energy_quadratic_default(self, generic_fixed):
        z = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        E = generic_fixed.compute_energy(z)
        expected = 0.5 * (1 + 4 + 9 + 16)
        assert E.item() == pytest.approx(expected, abs=1e-5)

    # -- Entropy --

    def test_compute_entropy_shape(self, generic_hard):
        torch.manual_seed(0)
        z = torch.randn(BATCH_SIZE, 4)
        S = generic_hard.compute_entropy(z)
        assert S.shape == (BATCH_SIZE,)

    def test_entropy_boltzmann_default(self, generic_fixed):
        """Non-learnable S: S(z) = -sum(|z_i| ln |z_i|)."""
        z = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        S = generic_fixed.compute_entropy(z)
        # -sum(1 * ln(1)) = 0
        assert S.item() == pytest.approx(0.0, abs=1e-5)

    # -- Gradients --

    def test_compute_gradients_shapes(self, generic_hard):
        torch.manual_seed(0)
        z = torch.randn(BATCH_SIZE, 4)
        grad_E, grad_S = generic_hard.compute_gradients(z)
        assert grad_E.shape == (BATCH_SIZE, 4)
        assert grad_S.shape == (BATCH_SIZE, 4)

    # -- project (GENERIC dynamics) --

    def test_project_output_shape(self, generic_hard):
        torch.manual_seed(0)
        z = torch.randn(BATCH_SIZE, 4)
        dz_dt = generic_hard.project(z)
        assert dz_dt.shape == (BATCH_SIZE, 4)

    def test_project_requires_state_dim(self):
        c = GENERICConstraint(mode="soft", state_dim=None)
        with pytest.raises(ValueError, match="state_dim required"):
            c.project(torch.randn(2, 4))

    # -- compute_violation --

    def test_violation_L_anti_symmetry_by_construction(self, generic_hard):
        """L = A - A^T is anti-symmetric by construction => ||L + L^T|| ~ 0."""
        L = generic_hard.get_L_matrix()
        anti_sym_err = torch.norm(L + L.T).item()
        assert anti_sym_err < 1e-5

    def test_violation_M_symmetry_by_construction(self, generic_hard):
        """M = B B^T is symmetric by construction => ||M - M^T|| ~ 0."""
        M = generic_hard.get_M_matrix()
        sym_err = torch.norm(M - M.T).item()
        assert sym_err < 1e-5

    def test_violation_nonneg(self, generic_hard):
        """compute_violation on a hard-mode constraint (which has structure initialized)."""
        torch.manual_seed(0)
        z = torch.randn(2, 5, 4)
        violation = generic_hard.compute_violation(z)
        assert violation.item() >= -1e-6

    def test_violation_without_state_dim_returns_zero(self):
        c = GENERICConstraint(mode="soft", state_dim=None)
        z = torch.randn(2, 5, 4)
        violation = c.compute_violation(z)
        assert violation.item() == 0.0

    def test_violation_uses_first_time_point(self, generic_hard):
        """compute_violation should handle 3D input by extracting z[:, 0, :]."""
        torch.manual_seed(0)
        z = torch.randn(2, 5, 4)
        violation = generic_hard.compute_violation(z)
        assert isinstance(violation, torch.Tensor)
        assert violation.ndim == 0

    def test_violation_handles_2d_input(self, generic_hard):
        """compute_violation should also handle 2D input."""
        torch.manual_seed(0)
        z = torch.randn(2, 4)
        violation = generic_hard.compute_violation(z)
        assert isinstance(violation, torch.Tensor)
        assert violation.ndim == 0

    # -- forward() --

    def test_forward_hard_output_types(self, generic_hard):
        torch.manual_seed(0)
        z = torch.randn(BATCH_SIZE, 4)
        z_out, violation = generic_hard(z)
        assert isinstance(z_out, torch.Tensor)
        assert isinstance(violation, torch.Tensor)
        assert z_out.shape == (BATCH_SIZE, 4)
        assert violation.item() == 0.0

    def test_forward_soft_output_types(self, generic_soft_no_dim):
        """Soft mode without initialized structure returns z unchanged and
        zero violation (since state_dim is None)."""
        torch.manual_seed(0)
        z = torch.randn(2, 5, 4)
        z_out, violation = generic_soft_no_dim(z)
        torch.testing.assert_close(z_out, z)
        assert isinstance(violation, torch.Tensor)
        assert violation.item() == pytest.approx(0.0, abs=1e-8)


# ===========================================================================
# ConstraintPipeline tests
# ===========================================================================


class TestConstraintPipeline:
    """Tests for the ConstraintPipeline."""

    def test_empty_pipeline_passthrough(self):
        pipeline = ConstraintPipeline([])
        z = _seeded_randn(BATCH_SIZE, STATE_DIM)
        z_out, violations = pipeline(z)
        torch.testing.assert_close(z_out, z)
        assert len(violations) == 0

    def test_single_hard_constraint(self):
        pos = PositivityConstraint(mode="hard", method="relu")
        pipeline = ConstraintPipeline([pos])
        z = _seeded_randn(BATCH_SIZE, STATE_DIM)
        z_out, violations = pipeline(z)
        assert torch.all(z_out >= 0)
        assert len(violations) == 0  # hard mode -> violation is 0

    def test_single_soft_constraint_with_violation(self):
        pos = PositivityConstraint(name="pos_soft", mode="soft", weight=1.0)
        pipeline = ConstraintPipeline([pos])
        z = torch.tensor([[-1.0, 2.0, -0.5]])
        _, violations = pipeline(z)
        assert isinstance(violations, dict)
        assert "pos_soft" in violations
        assert violations["pos_soft"].item() > 0

    def test_multiple_hard_constraints(self):
        pos = PositivityConstraint(name="positivity", mode="hard", method="relu")
        mb = MassBalanceConstraint(name="mass_balance", mode="hard")
        pipeline = ConstraintPipeline([pos, mb])
        z = torch.tensor([[-0.5, 1.5, 2.0]])
        z_out, violations = pipeline(z)
        assert torch.all(z_out >= 0)

    def test_mixed_hard_and_soft(self):
        pos_hard = PositivityConstraint(name="pos_hard", mode="hard", method="relu")
        pos_soft = PositivityConstraint(name="pos_soft", mode="soft", weight=1.0)
        pipeline = ConstraintPipeline([pos_hard, pos_soft])

        z = torch.tensor([[-1.0, 2.0, -0.5]])
        z_out, violations = pipeline(z)
        # Hard constraint makes everything non-negative first
        assert torch.all(z_out >= 0)
        # After hard projection, soft constraint sees all non-negative -> no violation
        assert "pos_soft" not in violations

    def test_pipeline_preserves_order(self):
        """Constraints applied in order: first positivity, then mass balance."""
        pos = PositivityConstraint(name="pos", mode="hard", method="relu")
        mb = MassBalanceConstraint(name="mb", mode="hard", check_total_mass=True)
        pipeline = ConstraintPipeline([pos, mb])

        z = torch.tensor([[-1.0, 3.0, 2.0]])
        z_out, _ = pipeline(z)
        # After relu: [epsilon, 3+eps, 2+eps]
        # After mass balance: scaled to preserve total mass from relu output
        assert z_out.shape == z.shape

    def test_returns_violations_dict_type(self):
        pos = PositivityConstraint(name="pos", mode="soft")
        pipeline = ConstraintPipeline([pos])
        z = torch.tensor([[-1.0, 2.0, 3.0]])
        _, violations = pipeline(z)
        assert isinstance(violations, dict)

    def test_repr(self):
        pos = PositivityConstraint(name="positivity")
        mb = MassBalanceConstraint(name="mass_balance")
        pipeline = ConstraintPipeline([pos, mb])
        r = repr(pipeline)
        assert "positivity" in r
        assert "mass_balance" in r

    def test_pipeline_is_nn_module(self):
        pos = PositivityConstraint(name="pos", mode="hard")
        pipeline = ConstraintPipeline([pos])
        assert isinstance(pipeline, torch.nn.Module)

    def test_pipeline_parameters_include_sub_constraints(self):
        """Pipeline's nn.ModuleList should expose sub-constraint parameters."""
        torch.manual_seed(0)
        ph = PortHamiltonianConstraint(mode="hard", state_dim=4)
        pipeline = ConstraintPipeline([ph])
        param_count = sum(p.numel() for p in pipeline.parameters())
        assert param_count > 0, "Pipeline should have learnable parameters from sub-constraints"

    def test_pipeline_three_constraints(self):
        """Pipeline with three constraints of different types."""
        pos = PositivityConstraint(name="pos", mode="hard", method="relu")
        mb = MassBalanceConstraint(name="mb", mode="hard")
        eb = EnergyBalanceConstraint(name="eb", mode="soft", weight=0.1)
        pipeline = ConstraintPipeline([pos, mb, eb])

        torch.manual_seed(0)
        z = torch.randn(2, 5, 4)
        z_out, violations = pipeline(z)
        assert z_out.shape == z.shape
        assert isinstance(violations, dict)


# ===========================================================================
# Cross-cutting / integration tests
# ===========================================================================


class TestOutputShapesAndTypes:
    """Verify that all constraints produce correct shapes and types."""

    @pytest.mark.parametrize("method", ["softplus", "relu", "square"])
    def test_positivity_2d_shapes(self, method):
        c = PositivityConstraint(mode="hard", method=method)
        z = _seeded_randn(BATCH_SIZE, STATE_DIM)
        z_out, viol = c(z)
        assert z_out.shape == (BATCH_SIZE, STATE_DIM)
        assert viol.ndim == 0

    @pytest.mark.parametrize("method", ["softplus", "relu", "square"])
    def test_positivity_3d_shapes(self, method):
        c = PositivityConstraint(mode="hard", method=method)
        z = _seeded_randn(BATCH_SIZE, TIME_STEPS, STATE_DIM)
        z_out, viol = c(z)
        assert z_out.shape == (BATCH_SIZE, TIME_STEPS, STATE_DIM)
        assert viol.ndim == 0

    def test_mass_balance_2d_shape(self):
        c = MassBalanceConstraint(mode="hard")
        z = _seeded_randn(BATCH_SIZE, STATE_DIM).abs()
        z_out, viol = c(z)
        assert z_out.shape == (BATCH_SIZE, STATE_DIM)
        assert viol.ndim == 0

    def test_energy_balance_3d_shape(self):
        c = EnergyBalanceConstraint(mode="soft")
        z = _seeded_randn(BATCH_SIZE, TIME_STEPS, STATE_DIM + 1)
        z_out, viol = c(z)
        assert z_out.shape == (BATCH_SIZE, TIME_STEPS, STATE_DIM + 1)
        assert viol.ndim == 0

    def test_thermodynamic_3d_shape(self):
        c = ThermodynamicConstraint(mode="soft")
        torch.manual_seed(0)
        z = torch.randn(BATCH_SIZE, TIME_STEPS, STATE_DIM + 1).abs() + 0.01
        z_out, viol = c(z)
        assert z_out.shape == (BATCH_SIZE, TIME_STEPS, STATE_DIM + 1)
        assert viol.ndim == 0

    def test_stoichiometric_2d_shape(self):
        S = torch.tensor([[-1.0, 1.0, 0.0]])
        c = StoichiometricConstraint(stoich_matrix=S, mode="hard")
        z = _seeded_randn(BATCH_SIZE, 3)
        z_out, viol = c(z)
        assert z_out.shape == (BATCH_SIZE, 3)
        assert viol.ndim == 0

    def test_port_hamiltonian_2d_shape(self):
        torch.manual_seed(0)
        c = PortHamiltonianConstraint(mode="hard", state_dim=4)
        z = torch.randn(BATCH_SIZE, 4)
        z_out, viol = c(z)
        assert z_out.shape == (BATCH_SIZE, 4)
        assert viol.ndim == 0

    def test_generic_2d_shape(self):
        torch.manual_seed(0)
        c = GENERICConstraint(mode="hard", state_dim=4)
        z = torch.randn(BATCH_SIZE, 4)
        z_out, viol = c(z)
        assert z_out.shape == (BATCH_SIZE, 4)
        assert viol.ndim == 0


class TestGradientFlow:
    """Verify that gradients flow through constraint operations."""

    def test_positivity_hard_gradient(self):
        c = PositivityConstraint(mode="hard", method="softplus")
        z = torch.randn(2, 3, requires_grad=True)
        z_out, _ = c(z)
        loss = z_out.sum()
        loss.backward()
        assert z.grad is not None
        assert z.grad.shape == z.shape

    def test_positivity_soft_gradient(self):
        c = PositivityConstraint(mode="soft", weight=1.0)
        z = torch.randn(2, 3, requires_grad=True)
        _, violation = c(z)
        violation.backward()
        assert z.grad is not None

    def test_stoichiometric_forward_gradient(self):
        S = torch.tensor([[-1.0, 1.0]])
        c = StoichiometricConstraint(stoich_matrix=S, mode="hard")
        rates = torch.randn(2, 1, requires_grad=True)
        dC_dt = c.forward_stoichiometry(rates)
        loss = dC_dt.sum()
        loss.backward()
        assert rates.grad is not None

    def test_port_hamiltonian_gradient(self):
        torch.manual_seed(0)
        c = PortHamiltonianConstraint(mode="hard", state_dim=4)
        z = torch.randn(2, 4, requires_grad=True)
        dz_dt = c.project(z)
        loss = dz_dt.sum()
        loss.backward()
        assert z.grad is not None

    def test_generic_gradient(self):
        torch.manual_seed(0)
        c = GENERICConstraint(mode="hard", state_dim=4)
        z = torch.randn(2, 4, requires_grad=True)
        dz_dt = c.project(z)
        loss = dz_dt.sum()
        loss.backward()
        assert z.grad is not None
