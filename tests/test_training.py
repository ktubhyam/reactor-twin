"""Tests for training infrastructure: data generation, losses, and trainer."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
from torch import nn

from reactor_twin.core.neural_ode import NeuralODE
from reactor_twin.physics.positivity import PositivityConstraint
from reactor_twin.reactors.cstr import CSTRReactor
from reactor_twin.reactors.systems import create_exothermic_cstr
from reactor_twin.training.data_generator import ReactorDataGenerator
from reactor_twin.training.losses import MultiObjectiveLoss
from reactor_twin.training.trainer import Trainer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
NUM_TIMES = 10
STATE_DIM = 2  # isothermal CSTR with 2 species


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isothermal_cstr():
    """Create an isothermal exothermic CSTR (state_dim=2: C_A, C_B)."""
    return create_exothermic_cstr(isothermal=True)


@pytest.fixture
def simple_cstr():
    """Create a simple isothermal CSTR without kinetics for fast tests."""
    return CSTRReactor(
        name="test_cstr",
        num_species=2,
        params={
            "V": 100.0,
            "F": 10.0,
            "C_feed": [1.0, 0.0],
            "T_feed": 350.0,
        },
        isothermal=True,
    )


@pytest.fixture
def data_generator(simple_cstr):
    """Create a reactor data generator with simple CSTR."""
    return ReactorDataGenerator(reactor=simple_cstr, method="LSODA")


@pytest.fixture
def exothermic_data_generator(isothermal_cstr):
    """Create a reactor data generator with exothermic CSTR."""
    return ReactorDataGenerator(reactor=isothermal_cstr, method="LSODA")


@pytest.fixture
def t_eval():
    """Common time evaluation array."""
    return np.linspace(0.0, 1.0, NUM_TIMES)


@pytest.fixture
def t_span():
    """Common time span tuple."""
    return (0.0, 1.0)


@pytest.fixture
def multi_loss():
    """Create a basic multi-objective loss with data and physics weights."""
    return MultiObjectiveLoss(weights={"data": 1.0, "physics": 0.1})


@pytest.fixture
def neural_ode_model():
    """Create a small NeuralODE for testing (state_dim=2, euler solver, no adjoint)."""
    return NeuralODE(
        state_dim=STATE_DIM,
        solver="euler",
        atol=1e-3,
        rtol=1e-2,
        adjoint=False,
        hidden_dim=16,
        num_layers=2,
    )


@pytest.fixture
def sample_predictions():
    """Sample predictions tensor for loss tests."""
    return torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)


@pytest.fixture
def sample_targets():
    """Sample targets tensor for loss tests."""
    return torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)


@pytest.fixture
def sample_batch(t_eval):
    """Create a synthetic batch dict for trainer tests."""
    z0 = torch.randn(BATCH_SIZE, STATE_DIM).abs()
    t_span_tensor = torch.tensor(t_eval, dtype=torch.float32)
    targets = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)
    return {
        "z0": z0,
        "t_span": t_span_tensor,
        "targets": targets,
    }


# ===========================================================================
# ReactorDataGenerator Tests
# ===========================================================================


class TestReactorDataGeneratorInit:
    """Tests for ReactorDataGenerator initialization."""

    def test_init_stores_reactor(self, simple_cstr):
        gen = ReactorDataGenerator(reactor=simple_cstr)
        assert gen.reactor is simple_cstr

    def test_init_default_method(self, simple_cstr):
        gen = ReactorDataGenerator(reactor=simple_cstr)
        assert gen.method == "LSODA"

    def test_init_custom_method(self, simple_cstr):
        gen = ReactorDataGenerator(reactor=simple_cstr, method="RK45")
        assert gen.method == "RK45"

    def test_init_default_tolerances(self, simple_cstr):
        gen = ReactorDataGenerator(reactor=simple_cstr)
        assert gen.rtol == 1e-6
        assert gen.atol == 1e-8

    def test_init_custom_tolerances(self, simple_cstr):
        gen = ReactorDataGenerator(reactor=simple_cstr, rtol=1e-4, atol=1e-6)
        assert gen.rtol == 1e-4
        assert gen.atol == 1e-6


class TestGenerateTrajectory:
    """Tests for generate_trajectory method."""

    def test_returns_dict_with_expected_keys(self, data_generator, t_span, t_eval):
        result = data_generator.generate_trajectory(t_span, t_eval)
        assert isinstance(result, dict)
        assert "t" in result
        assert "y" in result
        assert "success" in result

    def test_success_flag_is_true(self, data_generator, t_span, t_eval):
        result = data_generator.generate_trajectory(t_span, t_eval)
        assert result["success"] is True

    def test_time_array_shape(self, data_generator, t_span, t_eval):
        result = data_generator.generate_trajectory(t_span, t_eval)
        assert result["t"].shape == (NUM_TIMES,)

    def test_state_array_shape(self, data_generator, t_span, t_eval):
        result = data_generator.generate_trajectory(t_span, t_eval)
        assert result["y"].shape == (NUM_TIMES, STATE_DIM)

    def test_returns_numpy_arrays(self, data_generator, t_span, t_eval):
        result = data_generator.generate_trajectory(t_span, t_eval)
        assert isinstance(result["t"], np.ndarray)
        assert isinstance(result["y"], np.ndarray)

    def test_default_initial_condition(self, data_generator, t_span, t_eval):
        """When y0=None, should use reactor's default initial state."""
        result = data_generator.generate_trajectory(t_span, t_eval, y0=None)
        assert result["success"] is True
        expected_y0 = data_generator.reactor.get_initial_state()
        np.testing.assert_allclose(result["y"][0], expected_y0, rtol=1e-3)

    def test_custom_initial_condition(self, data_generator, t_span, t_eval):
        """Should start from the provided y0."""
        y0 = np.array([0.8, 0.2])
        result = data_generator.generate_trajectory(t_span, t_eval, y0=y0)
        assert result["success"] is True
        np.testing.assert_allclose(result["y"][0], y0, rtol=1e-3)

    def test_trajectory_values_are_finite(self, data_generator, t_span, t_eval):
        result = data_generator.generate_trajectory(t_span, t_eval)
        assert np.all(np.isfinite(result["y"]))

    def test_time_values_match_t_eval(self, data_generator, t_span, t_eval):
        result = data_generator.generate_trajectory(t_span, t_eval)
        np.testing.assert_allclose(result["t"], t_eval)

    def test_different_t_span_length(self, data_generator):
        """Should work with different numbers of time points."""
        t_eval_short = np.linspace(0.0, 1.0, 5)
        result = data_generator.generate_trajectory((0.0, 1.0), t_eval_short)
        assert result["y"].shape == (5, STATE_DIM)

    def test_with_exothermic_cstr(self, exothermic_data_generator, t_span, t_eval):
        """Should work with the create_exothermic_cstr factory (isothermal)."""
        result = exothermic_data_generator.generate_trajectory(t_span, t_eval)
        assert result["success"] is True
        assert result["y"].shape == (NUM_TIMES, STATE_DIM)


class TestGenerateBatch:
    """Tests for generate_batch method."""

    def test_returns_dict_with_expected_keys(self, data_generator, t_span, t_eval):
        batch = data_generator.generate_batch(BATCH_SIZE, t_span, t_eval)
        assert "z0" in batch
        assert "t_span" in batch
        assert "targets" in batch

    def test_z0_shape(self, data_generator, t_span, t_eval):
        batch = data_generator.generate_batch(BATCH_SIZE, t_span, t_eval)
        assert batch["z0"].shape == (BATCH_SIZE, STATE_DIM)

    def test_t_span_shape(self, data_generator, t_span, t_eval):
        batch = data_generator.generate_batch(BATCH_SIZE, t_span, t_eval)
        assert batch["t_span"].shape == (NUM_TIMES,)

    def test_targets_shape(self, data_generator, t_span, t_eval):
        batch = data_generator.generate_batch(BATCH_SIZE, t_span, t_eval)
        assert batch["targets"].shape == (BATCH_SIZE, NUM_TIMES, STATE_DIM)

    def test_returns_torch_tensors(self, data_generator, t_span, t_eval):
        batch = data_generator.generate_batch(BATCH_SIZE, t_span, t_eval)
        assert isinstance(batch["z0"], torch.Tensor)
        assert isinstance(batch["t_span"], torch.Tensor)
        assert isinstance(batch["targets"], torch.Tensor)

    def test_dtype_is_float32(self, data_generator, t_span, t_eval):
        batch = data_generator.generate_batch(BATCH_SIZE, t_span, t_eval)
        assert batch["z0"].dtype == torch.float32
        assert batch["t_span"].dtype == torch.float32
        assert batch["targets"].dtype == torch.float32

    def test_targets_are_finite(self, data_generator, t_span, t_eval):
        batch = data_generator.generate_batch(BATCH_SIZE, t_span, t_eval)
        assert torch.all(torch.isfinite(batch["targets"]))

    def test_z0_state_dim_matches_targets(self, data_generator, t_span, t_eval):
        batch = data_generator.generate_batch(BATCH_SIZE, t_span, t_eval)
        assert batch["z0"].shape[-1] == batch["targets"].shape[-1]

    def test_custom_initial_conditions(self, data_generator, t_span, t_eval):
        ic = np.array([[0.8, 0.0], [0.9, 0.0], [1.0, 0.0], [0.7, 0.1]])
        batch = data_generator.generate_batch(
            BATCH_SIZE, t_span, t_eval, initial_conditions=ic
        )
        np.testing.assert_allclose(batch["z0"].numpy(), ic, rtol=1e-5)

    def test_batch_size_one(self, data_generator, t_span, t_eval):
        batch = data_generator.generate_batch(1, t_span, t_eval)
        assert batch["z0"].shape == (1, STATE_DIM)
        assert batch["targets"].shape == (1, NUM_TIMES, STATE_DIM)

    def test_default_ic_perturbation_non_negative(self, data_generator, t_span, t_eval):
        """Default ICs with random perturbation should be clipped to non-negative."""
        batch = data_generator.generate_batch(BATCH_SIZE, t_span, t_eval)
        assert torch.all(batch["z0"] >= 0)


class TestGenerateDataset:
    """Tests for generate_dataset method."""

    def test_returns_list_of_batches(self, data_generator, t_span, t_eval):
        dataset = data_generator.generate_dataset(
            num_trajectories=8, t_span=t_span, t_eval=t_eval, batch_size=4
        )
        assert isinstance(dataset, list)

    def test_correct_number_of_batches_exact_division(
        self, data_generator, t_span, t_eval
    ):
        dataset = data_generator.generate_dataset(
            num_trajectories=8, t_span=t_span, t_eval=t_eval, batch_size=4
        )
        assert len(dataset) == 2

    def test_correct_number_of_batches_with_remainder(
        self, data_generator, t_span, t_eval
    ):
        """When num_trajectories is not divisible by batch_size, should ceil."""
        dataset = data_generator.generate_dataset(
            num_trajectories=7, t_span=t_span, t_eval=t_eval, batch_size=4
        )
        # ceil(7/4) = 2
        assert len(dataset) == 2

    def test_last_batch_correct_size_with_remainder(
        self, data_generator, t_span, t_eval
    ):
        """Last batch should have the remaining trajectories."""
        dataset = data_generator.generate_dataset(
            num_trajectories=7, t_span=t_span, t_eval=t_eval, batch_size=4
        )
        # First batch: 4 trajectories, second batch: 3 trajectories
        assert dataset[0]["z0"].shape[0] == 4
        assert dataset[1]["z0"].shape[0] == 3

    def test_each_batch_has_expected_keys(self, data_generator, t_span, t_eval):
        dataset = data_generator.generate_dataset(
            num_trajectories=8, t_span=t_span, t_eval=t_eval, batch_size=4
        )
        for batch in dataset:
            assert "z0" in batch
            assert "t_span" in batch
            assert "targets" in batch

    def test_each_batch_contains_tensors(self, data_generator, t_span, t_eval):
        dataset = data_generator.generate_dataset(
            num_trajectories=8, t_span=t_span, t_eval=t_eval, batch_size=4
        )
        for batch in dataset:
            assert isinstance(batch["z0"], torch.Tensor)
            assert isinstance(batch["t_span"], torch.Tensor)
            assert isinstance(batch["targets"], torch.Tensor)

    def test_single_trajectory_dataset(self, data_generator, t_span, t_eval):
        dataset = data_generator.generate_dataset(
            num_trajectories=1, t_span=t_span, t_eval=t_eval, batch_size=4
        )
        assert len(dataset) == 1
        assert dataset[0]["z0"].shape[0] == 1

    def test_total_trajectories_across_batches(self, data_generator, t_span, t_eval):
        num_trajectories = 8
        batch_size = 4
        dataset = data_generator.generate_dataset(
            num_trajectories=num_trajectories,
            t_span=t_span,
            t_eval=t_eval,
            batch_size=batch_size,
        )
        total = sum(batch["z0"].shape[0] for batch in dataset)
        assert total == num_trajectories


# ===========================================================================
# MultiObjectiveLoss Tests
# ===========================================================================


class TestMultiObjectiveLossInit:
    """Tests for MultiObjectiveLoss initialization."""

    def test_default_weights(self):
        loss_fn = MultiObjectiveLoss()
        assert loss_fn.weights == {"data": 1.0}

    def test_custom_weights(self):
        weights = {"data": 2.0, "physics": 0.5, "regularization": 0.01}
        loss_fn = MultiObjectiveLoss(weights=weights)
        assert loss_fn.weights == weights

    def test_default_no_constraints(self):
        loss_fn = MultiObjectiveLoss()
        assert loss_fn.constraints == []

    def test_custom_constraints(self):
        constraint = PositivityConstraint(mode="soft", weight=1.0)
        loss_fn = MultiObjectiveLoss(constraints=[constraint])
        assert len(loss_fn.constraints) == 1
        assert loss_fn.constraints[0] is constraint

    def test_is_nn_module(self):
        loss_fn = MultiObjectiveLoss()
        assert isinstance(loss_fn, nn.Module)


class TestDataLoss:
    """Tests for data_loss method (MSE)."""

    def test_data_loss_is_mse(self, multi_loss):
        preds = torch.tensor([[[1.0, 2.0]]])
        targets = torch.tensor([[[3.0, 4.0]]])
        loss = multi_loss.data_loss(preds, targets)
        expected = torch.mean((preds - targets) ** 2)
        torch.testing.assert_close(loss, expected)

    def test_zero_loss_for_identical_inputs(self, multi_loss, sample_targets):
        loss = multi_loss.data_loss(sample_targets, sample_targets)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_loss_is_scalar(self, multi_loss, sample_predictions, sample_targets):
        loss = multi_loss.data_loss(sample_predictions, sample_targets)
        assert loss.ndim == 0

    def test_loss_is_non_negative(self, multi_loss, sample_predictions, sample_targets):
        loss = multi_loss.data_loss(sample_predictions, sample_targets)
        assert loss.item() >= 0.0

    def test_symmetric(self, multi_loss, sample_predictions, sample_targets):
        """MSE(a, b) == MSE(b, a)."""
        loss_ab = multi_loss.data_loss(sample_predictions, sample_targets)
        loss_ba = multi_loss.data_loss(sample_targets, sample_predictions)
        torch.testing.assert_close(loss_ab, loss_ba)

    def test_known_mse_value(self, multi_loss):
        """Manual check: MSE of [[1,0]] vs [[0,0]] should be 0.5."""
        preds = torch.tensor([[[1.0, 0.0]]])
        targets = torch.tensor([[[0.0, 0.0]]])
        loss = multi_loss.data_loss(preds, targets)
        # (1^2 + 0^2) / 2 = 0.5
        assert loss.item() == pytest.approx(0.5, abs=1e-6)


class TestPhysicsLoss:
    """Tests for physics_loss method (placeholder)."""

    def test_returns_zero(self, multi_loss, sample_predictions, sample_targets):
        loss = multi_loss.physics_loss(sample_predictions, sample_targets)
        assert loss.item() == pytest.approx(0.0)

    def test_is_scalar(self, multi_loss, sample_predictions, sample_targets):
        loss = multi_loss.physics_loss(sample_predictions, sample_targets)
        assert loss.ndim == 0


class TestConstraintLoss:
    """Tests for constraint_loss method."""

    def test_empty_constraints_returns_empty_dict(
        self, sample_predictions
    ):
        loss_fn = MultiObjectiveLoss(constraints=[])
        result = loss_fn.constraint_loss(sample_predictions)
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_positivity_constraint_no_violation_on_positive_data(self):
        """All-positive predictions should produce no violation entry."""
        constraint = PositivityConstraint(mode="soft", weight=1.0)
        loss_fn = MultiObjectiveLoss(constraints=[constraint])
        preds = torch.abs(torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)) + 0.1
        result = loss_fn.constraint_loss(preds)
        # No violation means either empty dict or zero value
        if "positivity" in result:
            assert result["positivity"].item() == pytest.approx(0.0, abs=1e-6)

    def test_positivity_constraint_violation_on_negative_data(self):
        """Negative predictions should produce a non-zero positivity violation."""
        constraint = PositivityConstraint(mode="soft", weight=1.0)
        loss_fn = MultiObjectiveLoss(constraints=[constraint])
        preds = -torch.ones(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        result = loss_fn.constraint_loss(preds)
        assert "positivity" in result
        assert result["positivity"].item() > 0.0

    def test_positivity_hard_mode_returns_zero_violation(self):
        """Hard constraints project rather than penalize, so violation is 0."""
        constraint = PositivityConstraint(mode="hard", weight=1.0)
        loss_fn = MultiObjectiveLoss(constraints=[constraint])
        preds = -torch.ones(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        result = loss_fn.constraint_loss(preds)
        # Hard mode returns violation=0
        assert len(result) == 0

    def test_multiple_constraints(self):
        """Should aggregate violations from multiple constraints."""
        c1 = PositivityConstraint(name="pos_conc", mode="soft", weight=1.0, indices=[0])
        c2 = PositivityConstraint(name="pos_temp", mode="soft", weight=1.0, indices=[1])
        loss_fn = MultiObjectiveLoss(constraints=[c1, c2])
        preds = -torch.ones(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        result = loss_fn.constraint_loss(preds)
        assert "pos_conc" in result
        assert "pos_temp" in result


class TestRegularizationLoss:
    """Tests for regularization_loss method."""

    def test_returns_scalar(self, multi_loss, neural_ode_model):
        loss = multi_loss.regularization_loss(neural_ode_model)
        assert loss.ndim == 0

    def test_non_negative(self, multi_loss, neural_ode_model):
        loss = multi_loss.regularization_loss(neural_ode_model)
        assert loss.item() >= 0.0

    def test_is_sum_of_squared_norms(self, multi_loss):
        """L2 penalty should be sum of squared parameter norms."""
        model = nn.Linear(3, 2)
        loss = multi_loss.regularization_loss(model)
        expected = sum(torch.norm(p) ** 2 for p in model.parameters())
        torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-5)

    def test_zero_for_zero_model(self, multi_loss):
        """Model with all-zero params should have zero regularization."""
        model = nn.Linear(3, 2, bias=False)
        nn.init.zeros_(model.weight)
        loss = multi_loss.regularization_loss(model)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)


class TestMultiObjectiveLossForward:
    """Tests for forward method (total weighted loss)."""

    def test_returns_dict(self, multi_loss, sample_predictions, sample_targets):
        losses = multi_loss(sample_predictions, sample_targets)
        assert isinstance(losses, dict)

    def test_contains_total_key(self, multi_loss, sample_predictions, sample_targets):
        losses = multi_loss(sample_predictions, sample_targets)
        assert "total" in losses

    def test_contains_data_key(self, multi_loss, sample_predictions, sample_targets):
        losses = multi_loss(sample_predictions, sample_targets)
        assert "data" in losses

    def test_contains_physics_key(self, multi_loss, sample_predictions, sample_targets):
        losses = multi_loss(sample_predictions, sample_targets)
        assert "physics" in losses

    def test_total_is_scalar(self, multi_loss, sample_predictions, sample_targets):
        losses = multi_loss(sample_predictions, sample_targets)
        assert losses["total"].ndim == 0

    def test_total_is_weighted_sum_data_only(self, sample_predictions, sample_targets):
        """With only data weight, total = weight * data_loss."""
        loss_fn = MultiObjectiveLoss(weights={"data": 2.0})
        losses = loss_fn(sample_predictions, sample_targets)
        expected_total = 2.0 * losses["data"]
        torch.testing.assert_close(
            losses["total"], expected_total, atol=1e-5, rtol=1e-5
        )

    def test_total_includes_physics_weight(self, sample_predictions, sample_targets):
        """Physics weight should be applied (physics loss is 0 so total = data)."""
        loss_fn = MultiObjectiveLoss(weights={"data": 1.0, "physics": 10.0})
        losses = loss_fn(sample_predictions, sample_targets)
        # physics_loss returns 0, so total = 1.0 * data + 10.0 * 0 = data
        torch.testing.assert_close(
            losses["total"], losses["data"], atol=1e-5, rtol=1e-5
        )

    def test_regularization_included_when_weight_present(
        self, sample_predictions, sample_targets, neural_ode_model
    ):
        loss_fn = MultiObjectiveLoss(
            weights={"data": 1.0, "regularization": 0.01}
        )
        losses = loss_fn(sample_predictions, sample_targets, model=neural_ode_model)
        assert "regularization" in losses

    def test_no_regularization_without_weight(
        self, sample_predictions, sample_targets, neural_ode_model
    ):
        """If regularization not in weights, it should not be computed."""
        loss_fn = MultiObjectiveLoss(weights={"data": 1.0})
        losses = loss_fn(sample_predictions, sample_targets, model=neural_ode_model)
        assert "regularization" not in losses

    def test_no_regularization_without_model(
        self, sample_predictions, sample_targets
    ):
        """If model is None, regularization should not be computed."""
        loss_fn = MultiObjectiveLoss(
            weights={"data": 1.0, "regularization": 0.01}
        )
        losses = loss_fn(sample_predictions, sample_targets, model=None)
        assert "regularization" not in losses

    def test_constraint_violations_in_forward(self, sample_predictions, sample_targets):
        """Constraint violations should appear in the output losses dict."""
        constraint = PositivityConstraint(mode="soft", weight=1.0)
        loss_fn = MultiObjectiveLoss(
            weights={"data": 1.0, "positivity": 1.0},
            constraints=[constraint],
        )
        # Use negative predictions to trigger violation
        neg_preds = -torch.abs(sample_predictions)
        losses = loss_fn(neg_preds, sample_targets)
        assert "positivity" in losses
        assert losses["positivity"].item() > 0.0

    def test_zero_loss_for_identical_tensors(self, multi_loss, sample_targets):
        losses = multi_loss(sample_targets, sample_targets)
        assert losses["data"].item() == pytest.approx(0.0, abs=1e-7)

    def test_all_losses_are_tensors(self, multi_loss, sample_predictions, sample_targets):
        losses = multi_loss(sample_predictions, sample_targets)
        for key, val in losses.items():
            assert isinstance(val, torch.Tensor), f"Loss '{key}' is not a tensor"


class TestUpdateWeights:
    """Tests for update_weights method."""

    def test_updates_existing_weight(self, multi_loss):
        multi_loss.update_weights({"data": 5.0})
        assert multi_loss.weights["data"] == 5.0

    def test_adds_new_weight(self, multi_loss):
        multi_loss.update_weights({"new_term": 0.5})
        assert multi_loss.weights["new_term"] == 0.5

    def test_preserves_existing_weights_on_partial_update(self, multi_loss):
        original_physics = multi_loss.weights.get("physics", 0.1)
        multi_loss.update_weights({"data": 3.0})
        assert multi_loss.weights["physics"] == original_physics

    def test_update_affects_forward_computation(self, sample_predictions, sample_targets):
        loss_fn = MultiObjectiveLoss(weights={"data": 1.0})
        losses_before = loss_fn(sample_predictions, sample_targets)
        total_before = losses_before["total"].item()

        loss_fn.update_weights({"data": 10.0})
        losses_after = loss_fn(sample_predictions, sample_targets)
        total_after = losses_after["total"].item()

        # Total should be ~10x larger after weight update
        assert total_after == pytest.approx(10.0 * total_before, rel=1e-4)


# ===========================================================================
# Trainer Tests
# ===========================================================================


class TestTrainerInit:
    """Tests for Trainer initialization."""

    def test_stores_model(self, neural_ode_model, data_generator):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        assert trainer.model is neural_ode_model

    def test_stores_data_generator(self, neural_ode_model, data_generator):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        assert trainer.data_generator is data_generator

    def test_creates_default_loss_fn(self, neural_ode_model, data_generator):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        assert isinstance(trainer.loss_fn, MultiObjectiveLoss)

    def test_uses_provided_loss_fn(self, neural_ode_model, data_generator, multi_loss):
        trainer = Trainer(
            model=neural_ode_model,
            data_generator=data_generator,
            loss_fn=multi_loss,
        )
        assert trainer.loss_fn is multi_loss

    def test_creates_default_optimizer(self, neural_ode_model, data_generator):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        assert isinstance(trainer.optimizer, torch.optim.Adam)

    def test_uses_provided_optimizer(self, neural_ode_model, data_generator):
        optimizer = torch.optim.SGD(neural_ode_model.parameters(), lr=0.01)
        trainer = Trainer(
            model=neural_ode_model,
            data_generator=data_generator,
            optimizer=optimizer,
        )
        assert trainer.optimizer is optimizer

    def test_default_device_is_cpu(self, neural_ode_model, data_generator):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        assert trainer.device == torch.device("cpu")

    def test_initial_epoch_is_zero(self, neural_ode_model, data_generator):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        assert trainer.epoch == 0

    def test_initial_global_step_is_zero(self, neural_ode_model, data_generator):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        assert trainer.global_step == 0

    def test_initial_best_val_loss_is_inf(self, neural_ode_model, data_generator):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        assert trainer.best_val_loss == float("inf")

    def test_initial_history_is_empty(self, neural_ode_model, data_generator):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        assert trainer.history == {"train_loss": [], "val_loss": []}

    def test_scheduler_stored(self, neural_ode_model, data_generator):
        optimizer = torch.optim.Adam(neural_ode_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
        trainer = Trainer(
            model=neural_ode_model,
            data_generator=data_generator,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        assert trainer.scheduler is scheduler

    def test_no_scheduler_by_default(self, neural_ode_model, data_generator):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        assert trainer.scheduler is None


class TestTrainEpoch:
    """Tests for train_epoch method."""

    def test_returns_dict_of_losses(self, neural_ode_model, data_generator, sample_batch):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        avg_losses = trainer.train_epoch([sample_batch])
        assert isinstance(avg_losses, dict)

    def test_contains_total_loss(self, neural_ode_model, data_generator, sample_batch):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        avg_losses = trainer.train_epoch([sample_batch])
        assert "total" in avg_losses

    def test_total_loss_is_float(self, neural_ode_model, data_generator, sample_batch):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        avg_losses = trainer.train_epoch([sample_batch])
        assert isinstance(avg_losses["total"], float)

    def test_increments_global_step(self, neural_ode_model, data_generator, sample_batch):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        assert trainer.global_step == 0
        trainer.train_epoch([sample_batch, sample_batch])
        assert trainer.global_step == 2

    def test_model_in_train_mode_after_epoch(
        self, neural_ode_model, data_generator, sample_batch
    ):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        trainer.train_epoch([sample_batch])
        assert neural_ode_model.training

    def test_multiple_batches(self, neural_ode_model, data_generator, sample_batch):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        avg_losses = trainer.train_epoch([sample_batch, sample_batch, sample_batch])
        assert isinstance(avg_losses["total"], float)
        assert trainer.global_step == 3

    def test_loss_is_finite(self, neural_ode_model, data_generator, sample_batch):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        avg_losses = trainer.train_epoch([sample_batch])
        assert np.isfinite(avg_losses["total"])


class TestValidate:
    """Tests for validate method."""

    def test_returns_dict_of_losses(self, neural_ode_model, data_generator, sample_batch):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        val_losses = trainer.validate([sample_batch])
        assert isinstance(val_losses, dict)

    def test_contains_total_loss(self, neural_ode_model, data_generator, sample_batch):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        val_losses = trainer.validate([sample_batch])
        assert "total" in val_losses

    def test_total_loss_is_float(self, neural_ode_model, data_generator, sample_batch):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        val_losses = trainer.validate([sample_batch])
        assert isinstance(val_losses["total"], float)

    def test_does_not_change_global_step(
        self, neural_ode_model, data_generator, sample_batch
    ):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        step_before = trainer.global_step
        trainer.validate([sample_batch])
        assert trainer.global_step == step_before

    def test_model_in_eval_mode_during_validate(
        self, neural_ode_model, data_generator, sample_batch
    ):
        """After validate, model should be in eval mode."""
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        trainer.validate([sample_batch])
        assert not neural_ode_model.training

    def test_loss_is_finite(self, neural_ode_model, data_generator, sample_batch):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        val_losses = trainer.validate([sample_batch])
        assert np.isfinite(val_losses["total"])


class TestTrainLoop:
    """Tests for the full train method."""

    def test_returns_history(self, neural_ode_model, data_generator):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        t_eval = np.linspace(0.0, 1.0, NUM_TIMES)
        history = trainer.train(
            num_epochs=2,
            t_span=(0.0, 1.0),
            t_eval=t_eval,
            train_trajectories=8,
            val_trajectories=4,
            batch_size=4,
            val_interval=1,
        )
        assert isinstance(history, dict)

    def test_history_has_train_loss(self, neural_ode_model, data_generator):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        t_eval = np.linspace(0.0, 1.0, NUM_TIMES)
        history = trainer.train(
            num_epochs=2,
            t_span=(0.0, 1.0),
            t_eval=t_eval,
            train_trajectories=8,
            val_trajectories=4,
            batch_size=4,
        )
        assert "train_loss" in history
        assert len(history["train_loss"]) == 2

    def test_history_has_val_loss(self, neural_ode_model, data_generator):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        t_eval = np.linspace(0.0, 1.0, NUM_TIMES)
        history = trainer.train(
            num_epochs=2,
            t_span=(0.0, 1.0),
            t_eval=t_eval,
            train_trajectories=8,
            val_trajectories=4,
            batch_size=4,
            val_interval=1,
        )
        assert "val_loss" in history
        # Validation every epoch with val_interval=1
        assert len(history["val_loss"]) == 2

    def test_epoch_counter_advances(self, neural_ode_model, data_generator):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        t_eval = np.linspace(0.0, 1.0, NUM_TIMES)
        trainer.train(
            num_epochs=2,
            t_span=(0.0, 1.0),
            t_eval=t_eval,
            train_trajectories=8,
            val_trajectories=4,
            batch_size=4,
        )
        assert trainer.epoch == 1  # 0-indexed, last epoch

    def test_val_interval_controls_validation_frequency(
        self, neural_ode_model, data_generator
    ):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        t_eval = np.linspace(0.0, 1.0, NUM_TIMES)
        history = trainer.train(
            num_epochs=4,
            t_span=(0.0, 1.0),
            t_eval=t_eval,
            train_trajectories=8,
            val_trajectories=4,
            batch_size=4,
            val_interval=2,
        )
        # val_interval=2 means validate at epochs 1, 3 (when (epoch+1)%2==0)
        assert len(history["val_loss"]) == 2


class TestCheckpoint:
    """Tests for save_checkpoint and load_checkpoint methods."""

    def test_save_creates_file(self, neural_ode_model, data_generator):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_checkpoint(tmpdir, "test_ckpt.pt")
            assert os.path.exists(os.path.join(tmpdir, "test_ckpt.pt"))

    def test_save_creates_directory(self, neural_ode_model, data_generator):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "subdir", "checkpoints")
            trainer.save_checkpoint(nested_dir, "test_ckpt.pt")
            assert os.path.exists(os.path.join(nested_dir, "test_ckpt.pt"))

    def test_load_restores_epoch(self, neural_ode_model, data_generator):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        trainer.epoch = 5
        trainer.global_step = 100

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_checkpoint(tmpdir, "ckpt.pt")

            # Create a new trainer and load checkpoint
            trainer2 = Trainer(model=neural_ode_model, data_generator=data_generator)
            assert trainer2.epoch == 0
            trainer2.load_checkpoint(os.path.join(tmpdir, "ckpt.pt"))
            assert trainer2.epoch == 5

    def test_load_restores_global_step(self, neural_ode_model, data_generator):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        trainer.global_step = 42

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_checkpoint(tmpdir, "ckpt.pt")

            trainer2 = Trainer(model=neural_ode_model, data_generator=data_generator)
            trainer2.load_checkpoint(os.path.join(tmpdir, "ckpt.pt"))
            assert trainer2.global_step == 42

    def test_load_restores_best_val_loss(self, neural_ode_model, data_generator):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        trainer.best_val_loss = 0.123

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_checkpoint(tmpdir, "ckpt.pt")

            trainer2 = Trainer(model=neural_ode_model, data_generator=data_generator)
            trainer2.load_checkpoint(os.path.join(tmpdir, "ckpt.pt"))
            assert trainer2.best_val_loss == pytest.approx(0.123)

    def test_load_restores_history(self, neural_ode_model, data_generator):
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        trainer.history = {"train_loss": [1.0, 0.5, 0.25], "val_loss": [1.1, 0.6]}

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_checkpoint(tmpdir, "ckpt.pt")

            trainer2 = Trainer(model=neural_ode_model, data_generator=data_generator)
            trainer2.load_checkpoint(os.path.join(tmpdir, "ckpt.pt"))
            assert trainer2.history == {
                "train_loss": [1.0, 0.5, 0.25],
                "val_loss": [1.1, 0.6],
            }

    def test_roundtrip_preserves_model_weights(self, data_generator):
        """Save and load should preserve model parameters."""
        model = NeuralODE(
            state_dim=STATE_DIM,
            solver="euler",
            adjoint=False,
            hidden_dim=16,
            num_layers=2,
        )
        trainer = Trainer(model=model, data_generator=data_generator)

        # Capture original weights
        original_state = {
            k: v.clone() for k, v in model.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_checkpoint(tmpdir, "ckpt.pt")

            # Create a fresh model and trainer
            model2 = NeuralODE(
                state_dim=STATE_DIM,
                solver="euler",
                adjoint=False,
                hidden_dim=16,
                num_layers=2,
            )
            trainer2 = Trainer(model=model2, data_generator=data_generator)

            trainer2.load_checkpoint(os.path.join(tmpdir, "ckpt.pt"))

            # Compare model parameters
            for key, val in original_state.items():
                torch.testing.assert_close(
                    model2.state_dict()[key],
                    val,
                    msg=f"Mismatch in parameter {key}",
                )

    def test_checkpoint_file_contents(self, neural_ode_model, data_generator):
        """Checkpoint should contain expected keys."""
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "ckpt.pt")
            trainer.save_checkpoint(tmpdir, "ckpt.pt")

            checkpoint = torch.load(ckpt_path, map_location="cpu")
            assert "epoch" in checkpoint
            assert "global_step" in checkpoint
            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint
            assert "best_val_loss" in checkpoint
            assert "history" in checkpoint


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestIntegration:
    """End-to-end integration tests combining data generator, loss, and trainer."""

    def test_generated_data_works_with_loss(self, data_generator, multi_loss):
        """Data from generator should be compatible with loss function."""
        t_eval = np.linspace(0.0, 1.0, NUM_TIMES)
        batch = data_generator.generate_batch(BATCH_SIZE, (0.0, 1.0), t_eval)

        # Use targets as both predictions and targets (sanity check)
        losses = multi_loss(batch["targets"], batch["targets"])
        assert losses["data"].item() == pytest.approx(0.0, abs=1e-6)

    def test_model_output_shape_matches_targets(
        self, neural_ode_model, data_generator
    ):
        """NeuralODE output should match the shape of generated targets."""
        t_eval = np.linspace(0.0, 1.0, NUM_TIMES)
        batch = data_generator.generate_batch(BATCH_SIZE, (0.0, 1.0), t_eval)

        predictions = neural_ode_model(
            z0=batch["z0"], t_span=batch["t_span"]
        )
        assert predictions.shape == batch["targets"].shape

    def test_train_epoch_with_generated_data(
        self, neural_ode_model, data_generator
    ):
        """Full pipeline: generate data -> train epoch -> get losses."""
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        t_eval = np.linspace(0.0, 1.0, NUM_TIMES)

        dataset = data_generator.generate_dataset(
            num_trajectories=8,
            t_span=(0.0, 1.0),
            t_eval=t_eval,
            batch_size=4,
        )

        avg_losses = trainer.train_epoch(dataset)
        assert np.isfinite(avg_losses["total"])
        assert avg_losses["total"] > 0

    def test_validate_with_generated_data(
        self, neural_ode_model, data_generator
    ):
        """Full pipeline: generate data -> validate -> get losses."""
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        t_eval = np.linspace(0.0, 1.0, NUM_TIMES)

        dataset = data_generator.generate_dataset(
            num_trajectories=4,
            t_span=(0.0, 1.0),
            t_eval=t_eval,
            batch_size=4,
        )

        val_losses = trainer.validate(dataset)
        assert np.isfinite(val_losses["total"])

    def test_full_train_loop_with_generated_data(
        self, neural_ode_model, data_generator
    ):
        """End-to-end: create trainer -> train for 2 epochs -> check history."""
        trainer = Trainer(model=neural_ode_model, data_generator=data_generator)
        t_eval = np.linspace(0.0, 1.0, NUM_TIMES)

        history = trainer.train(
            num_epochs=2,
            t_span=(0.0, 1.0),
            t_eval=t_eval,
            train_trajectories=8,
            val_trajectories=4,
            batch_size=4,
            val_interval=1,
        )

        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2
        assert all(np.isfinite(v) for v in history["train_loss"])
        assert all(np.isfinite(v) for v in history["val_loss"])

    def test_loss_with_constraint_and_model(self, neural_ode_model, data_generator):
        """Loss function with constraints and regularization on real model output."""
        constraint = PositivityConstraint(mode="soft", weight=1.0)
        loss_fn = MultiObjectiveLoss(
            weights={"data": 1.0, "positivity": 0.1, "regularization": 0.001},
            constraints=[constraint],
        )

        t_eval = np.linspace(0.0, 1.0, NUM_TIMES)
        batch = data_generator.generate_batch(BATCH_SIZE, (0.0, 1.0), t_eval)
        predictions = neural_ode_model(z0=batch["z0"], t_span=batch["t_span"])

        losses = loss_fn(predictions, batch["targets"], model=neural_ode_model)
        assert "total" in losses
        assert "data" in losses
        assert "regularization" in losses
        assert np.isfinite(losses["total"].item())

    def test_train_with_checkpoint(self, neural_ode_model, data_generator):
        """Train with checkpoint directory, ensure checkpoints are saved on validation improvement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=neural_ode_model, data_generator=data_generator
            )
            t_eval = np.linspace(0.0, 1.0, NUM_TIMES)
            trainer.train(
                num_epochs=2,
                t_span=(0.0, 1.0),
                t_eval=t_eval,
                train_trajectories=8,
                val_trajectories=4,
                batch_size=4,
                val_interval=1,
                checkpoint_dir=tmpdir,
            )
            # Best model should be saved since initial best_val_loss is inf
            assert os.path.exists(os.path.join(tmpdir, "best_model.pt"))


# ===========================================================================
# Data Generator Failure / Retry Tests
# ===========================================================================


class TestDataGeneratorFailurePaths:
    """Tests for integration failure logging and retry logic in data_generator.py."""

    def test_trajectory_failure_logs_warning(self, isothermal_cstr):
        """Cover line 91: when solve_ivp returns success=False, warning is logged."""
        from unittest.mock import MagicMock, patch

        gen = ReactorDataGenerator(isothermal_cstr)
        t_span = (0.0, 1.0)
        t_eval = np.linspace(0.0, 1.0, NUM_TIMES)

        # Mock solve_ivp to return a failed result
        mock_sol = MagicMock()
        mock_sol.success = False
        mock_sol.message = "Integration failed (test)"
        mock_sol.t = t_eval
        mock_sol.y = np.zeros((STATE_DIM, NUM_TIMES))

        with patch(
            "reactor_twin.training.data_generator.solve_ivp", return_value=mock_sol
        ):
            result = gen.generate_trajectory(t_span, t_eval)
            assert result["success"] is False

    def test_batch_retry_with_default_ic_on_failure(self, isothermal_cstr):
        """Cover lines 142-150: retry logic when trajectory generation fails.

        First call fails, retry with default IC succeeds.
        """
        from unittest.mock import patch

        gen = ReactorDataGenerator(isothermal_cstr)
        t_span = (0.0, 1.0)
        t_eval = np.linspace(0.0, 1.0, NUM_TIMES)

        # First call: fail. Second call (retry with default IC): succeed.
        call_count = 0

        def mock_generate_trajectory(t_span, t_eval, y0, controls=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: fail
                return {
                    "t": t_eval,
                    "y": np.zeros((NUM_TIMES, STATE_DIM)),
                    "success": False,
                }
            else:
                # Retry: succeed
                return {
                    "t": t_eval,
                    "y": np.ones((NUM_TIMES, STATE_DIM)) * 0.5,
                    "success": True,
                }

        with patch.object(gen, "generate_trajectory", side_effect=mock_generate_trajectory):
            batch = gen.generate_batch(1, t_span, t_eval)
            assert batch["targets"].shape == (1, NUM_TIMES, STATE_DIM)
            # The trajectory should be the retry result (0.5s), not zeros
            assert torch.all(batch["targets"] > 0)

    def test_batch_retry_both_fail_raises(self, isothermal_cstr):
        """When every trajectory fails (original + retry), RuntimeError is raised."""
        from unittest.mock import patch

        gen = ReactorDataGenerator(isothermal_cstr)
        t_span = (0.0, 1.0)
        t_eval = np.linspace(0.0, 1.0, NUM_TIMES)

        def always_fail(t_span, t_eval, y0, controls=None):
            return {
                "t": t_eval,
                "y": np.zeros((NUM_TIMES, STATE_DIM)),
                "success": False,
            }

        with patch.object(gen, "generate_trajectory", side_effect=always_fail):
            with pytest.raises(RuntimeError, match="All trajectories in batch failed"):
                gen.generate_batch(1, t_span, t_eval)
