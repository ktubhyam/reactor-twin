"""Tests for core Neural DE variants and ODE function networks."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from reactor_twin.core import (
    AugmentedNeuralODE,
    NeuralODE,
)
from reactor_twin.core.latent_neural_ode import (
    Decoder,
    Encoder,
    LatentNeuralODE,
)
from reactor_twin.core.ode_func import (
    HybridODEFunc,
    MLPODEFunc,
    PortHamiltonianODEFunc,
    ResNetODEFunc,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_DIM = 3
BATCH_SIZE = 4
NUM_TIMES = 10
LATENT_DIM = 4
AUGMENT_DIM = 2
INPUT_DIM = 2
HIDDEN_DIM = 32
SEQ_LEN = 5

# Use euler + adjoint=False everywhere for fast, deterministic tests.
SOLVER = "euler"
ADJOINT = False

# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def seed_everything():
    """Fix random seed for reproducibility in every test."""
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Fixtures — ODE functions
# ---------------------------------------------------------------------------


@pytest.fixture
def mlp_func():
    """MLPODEFunc with default (softplus) activation, no control input."""
    return MLPODEFunc(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, num_layers=2)


@pytest.fixture
def mlp_func_with_input():
    """MLPODEFunc that accepts external control inputs."""
    return MLPODEFunc(
        state_dim=STATE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=2,
        input_dim=INPUT_DIM,
    )


@pytest.fixture
def resnet_func():
    """ResNet ODE function."""
    return ResNetODEFunc(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, num_layers=2)


@pytest.fixture
def port_hamiltonian_func():
    """Port-Hamiltonian ODE function."""
    return PortHamiltonianODEFunc(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM)


# ---------------------------------------------------------------------------
# Fixtures — full models
# ---------------------------------------------------------------------------


@pytest.fixture
def neural_ode():
    """Standard NeuralODE (euler, no adjoint)."""
    return NeuralODE(
        state_dim=STATE_DIM,
        solver=SOLVER,
        atol=1e-3,
        rtol=1e-2,
        adjoint=ADJOINT,
        hidden_dim=HIDDEN_DIM,
        num_layers=2,
    )


@pytest.fixture
def neural_ode_with_output_dim():
    """NeuralODE with an explicit output_dim different from state_dim."""
    return NeuralODE(
        state_dim=STATE_DIM,
        solver=SOLVER,
        adjoint=ADJOINT,
        output_dim=5,
        hidden_dim=HIDDEN_DIM,
        num_layers=2,
    )


@pytest.fixture
def augmented_ode():
    """AugmentedNeuralODE (euler, no adjoint)."""
    return AugmentedNeuralODE(
        state_dim=STATE_DIM,
        augment_dim=AUGMENT_DIM,
        solver=SOLVER,
        atol=1e-3,
        rtol=1e-2,
        adjoint=ADJOINT,
        hidden_dim=HIDDEN_DIM,
        num_layers=2,
    )


@pytest.fixture
def latent_ode_gru():
    """LatentNeuralODE with GRU encoder (euler, no adjoint)."""
    return LatentNeuralODE(
        state_dim=STATE_DIM,
        latent_dim=LATENT_DIM,
        encoder_type="gru",
        solver=SOLVER,
        adjoint=ADJOINT,
        encoder_hidden_dim=HIDDEN_DIM,
        decoder_hidden_dim=HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=2,
    )


@pytest.fixture
def latent_ode_mlp():
    """LatentNeuralODE with MLP encoder (euler, no adjoint)."""
    return LatentNeuralODE(
        state_dim=STATE_DIM,
        latent_dim=LATENT_DIM,
        encoder_type="mlp",
        solver=SOLVER,
        adjoint=ADJOINT,
        encoder_hidden_dim=HIDDEN_DIM,
        decoder_hidden_dim=HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=2,
    )


# ===================================================================
# MLPODEFunc Tests
# ===================================================================


class TestMLPODEFunc:
    """Tests for MLP-based ODE function."""

    def test_forward_shape(self, mlp_func):
        t = torch.tensor(0.0)
        z = torch.randn(BATCH_SIZE, STATE_DIM)
        with torch.no_grad():
            dzdt = mlp_func(t, z)
        assert dzdt.shape == (BATCH_SIZE, STATE_DIM)

    def test_forward_with_scalar_time(self, mlp_func):
        t = torch.tensor(1.5)
        z = torch.randn(BATCH_SIZE, STATE_DIM)
        with torch.no_grad():
            dzdt = mlp_func(t, z)
        assert dzdt.shape == (BATCH_SIZE, STATE_DIM)

    def test_forward_with_batch_time(self, mlp_func):
        t = torch.randn(BATCH_SIZE)
        z = torch.randn(BATCH_SIZE, STATE_DIM)
        with torch.no_grad():
            dzdt = mlp_func(t, z)
        assert dzdt.shape == (BATCH_SIZE, STATE_DIM)

    def test_output_is_finite(self, mlp_func):
        t = torch.tensor(0.0)
        z = torch.randn(BATCH_SIZE, STATE_DIM)
        with torch.no_grad():
            dzdt = mlp_func(t, z)
        assert torch.all(torch.isfinite(dzdt))

    def test_network_input_size_without_controls(self):
        """input_size = state_dim + 1 (time) when input_dim=0."""
        func = MLPODEFunc(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, num_layers=2)
        first_layer = func.net[0]
        assert first_layer.in_features == STATE_DIM + 1

    def test_network_input_size_with_controls(self):
        """input_size = state_dim + input_dim + 1 when input_dim>0."""
        func = MLPODEFunc(
            state_dim=STATE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=2,
            input_dim=INPUT_DIM,
        )
        first_layer = func.net[0]
        assert first_layer.in_features == STATE_DIM + INPUT_DIM + 1

    def test_forward_with_controls(self, mlp_func_with_input):
        t = torch.tensor(0.0)
        z = torch.randn(BATCH_SIZE, STATE_DIM)
        u = torch.randn(BATCH_SIZE, INPUT_DIM)
        with torch.no_grad():
            dzdt = mlp_func_with_input(t, z, u=u)
        assert dzdt.shape == (BATCH_SIZE, STATE_DIM)

    def test_activation_softplus(self):
        func = MLPODEFunc(
            state_dim=STATE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=3,
            activation="softplus",
        )
        activations = [m for m in func.net.modules() if isinstance(m, nn.Softplus)]
        assert len(activations) == 2  # num_layers - 1

    def test_activation_tanh(self):
        func = MLPODEFunc(
            state_dim=STATE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=3,
            activation="tanh",
        )
        activations = [m for m in func.net.modules() if isinstance(m, nn.Tanh)]
        assert len(activations) == 2

    def test_activation_relu(self):
        func = MLPODEFunc(
            state_dim=STATE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=3,
            activation="relu",
        )
        activations = [m for m in func.net.modules() if isinstance(m, nn.ReLU)]
        assert len(activations) == 2

    def test_unknown_activation_raises(self):
        from reactor_twin.exceptions import ValidationError

        with pytest.raises(ValidationError, match="Unknown activation"):
            MLPODEFunc(
                state_dim=STATE_DIM,
                hidden_dim=HIDDEN_DIM,
                num_layers=3,
                activation="gelu",
            )

    def test_default_init_params(self):
        func = MLPODEFunc(state_dim=5)
        assert func.state_dim == 5
        assert func.input_dim == 0
        assert func.hidden_dim == 64
        assert func.num_layers == 3

    def test_gradient_flows(self, mlp_func):
        """Ensure gradients propagate through the MLP ODE function."""
        t = torch.tensor(0.0)
        z = torch.randn(BATCH_SIZE, STATE_DIM, requires_grad=True)
        dzdt = mlp_func(t, z)
        loss = dzdt.sum()
        loss.backward()
        assert z.grad is not None
        assert z.grad.shape == (BATCH_SIZE, STATE_DIM)


# ===================================================================
# ResNetODEFunc Tests
# ===================================================================


class TestResNetODEFunc:
    """Tests for ResNet ODE function."""

    def test_forward_shape(self, resnet_func):
        t = torch.tensor(0.0)
        z = torch.randn(BATCH_SIZE, STATE_DIM)
        with torch.no_grad():
            dzdt = resnet_func(t, z)
        assert dzdt.shape == (BATCH_SIZE, STATE_DIM)

    def test_output_is_finite(self, resnet_func):
        t = torch.tensor(0.0)
        z = torch.randn(BATCH_SIZE, STATE_DIM)
        with torch.no_grad():
            dzdt = resnet_func(t, z)
        assert torch.all(torch.isfinite(dzdt))

    def test_has_residual_blocks(self, resnet_func):
        assert len(resnet_func.blocks) == 2


# ===================================================================
# PortHamiltonianODEFunc Tests
# ===================================================================


class TestPortHamiltonianODEFunc:
    """Tests for Port-Hamiltonian ODE function."""

    def test_forward_shape(self, port_hamiltonian_func):
        t = torch.tensor(0.0)
        z = torch.randn(BATCH_SIZE, STATE_DIM)
        dzdt = port_hamiltonian_func(t, z)
        assert dzdt.shape == (BATCH_SIZE, STATE_DIM)

    def test_J_is_skew_symmetric(self, port_hamiltonian_func):
        J = port_hamiltonian_func.get_J()
        torch.testing.assert_close(J, -J.T, atol=1e-6, rtol=1e-6)

    def test_R_is_positive_semidefinite(self, port_hamiltonian_func):
        R = port_hamiltonian_func.get_R()
        eigenvalues = torch.linalg.eigvalsh(R)
        assert torch.all(eigenvalues >= -1e-6), "R matrix has negative eigenvalues"

    def test_hamiltonian_returns_scalar(self, port_hamiltonian_func):
        z = torch.randn(BATCH_SIZE, STATE_DIM)
        H = port_hamiltonian_func.hamiltonian(z)
        assert H.shape == (BATCH_SIZE,)


# ===================================================================
# HybridODEFunc Tests
# ===================================================================


class TestHybridODEFunc:
    """Tests for Hybrid physics + neural ODE function."""

    def test_forward_shape(self, mlp_func):
        physics = MLPODEFunc(state_dim=STATE_DIM, hidden_dim=16, num_layers=2)
        hybrid = HybridODEFunc(
            state_dim=STATE_DIM,
            physics_func=physics,
            neural_func=mlp_func,
        )
        t = torch.tensor(0.0)
        z = torch.randn(BATCH_SIZE, STATE_DIM)
        with torch.no_grad():
            dzdt = hybrid(t, z)
        assert dzdt.shape == (BATCH_SIZE, STATE_DIM)

    def test_combines_physics_and_neural_default_weight(self, mlp_func):
        """With correction_weight=1.0 (default): output = physics + neural."""
        physics = MLPODEFunc(state_dim=STATE_DIM, hidden_dim=16, num_layers=2)
        hybrid = HybridODEFunc(
            state_dim=STATE_DIM,
            physics_func=physics,
            neural_func=mlp_func,
            correction_weight=1.0,
        )
        t = torch.tensor(0.0)
        z = torch.randn(BATCH_SIZE, STATE_DIM)

        with torch.no_grad():
            physics_out = physics(t, z)
            neural_out = mlp_func(t, z)
            hybrid_out = hybrid(t, z)

        expected = physics_out + 1.0 * neural_out
        torch.testing.assert_close(hybrid_out, expected, atol=1e-5, rtol=1e-5)

    def test_combines_physics_and_neural_custom_weight(self, mlp_func):
        """With correction_weight=0.5: output = physics + 0.5 * neural."""
        physics = MLPODEFunc(state_dim=STATE_DIM, hidden_dim=16, num_layers=2)
        hybrid = HybridODEFunc(
            state_dim=STATE_DIM,
            physics_func=physics,
            neural_func=mlp_func,
            correction_weight=0.5,
        )
        t = torch.tensor(0.0)
        z = torch.randn(BATCH_SIZE, STATE_DIM)

        with torch.no_grad():
            physics_out = physics(t, z)
            neural_out = mlp_func(t, z)
            hybrid_out = hybrid(t, z)

        expected = physics_out + 0.5 * neural_out
        torch.testing.assert_close(hybrid_out, expected, atol=1e-5, rtol=1e-5)

    def test_zero_correction_weight_gives_pure_physics(self, mlp_func):
        """With correction_weight=0.0: output = physics only."""
        physics = MLPODEFunc(state_dim=STATE_DIM, hidden_dim=16, num_layers=2)
        hybrid = HybridODEFunc(
            state_dim=STATE_DIM,
            physics_func=physics,
            neural_func=mlp_func,
            correction_weight=0.0,
        )
        t = torch.tensor(0.0)
        z = torch.randn(BATCH_SIZE, STATE_DIM)

        with torch.no_grad():
            physics_out = physics(t, z)
            hybrid_out = hybrid(t, z)

        torch.testing.assert_close(hybrid_out, physics_out, atol=1e-6, rtol=1e-6)

    def test_hybrid_with_controls(self):
        """HybridODEFunc forwards controls to both components."""
        physics = MLPODEFunc(
            state_dim=STATE_DIM, hidden_dim=16, num_layers=2, input_dim=INPUT_DIM
        )
        neural = MLPODEFunc(
            state_dim=STATE_DIM, hidden_dim=16, num_layers=2, input_dim=INPUT_DIM
        )
        hybrid = HybridODEFunc(
            state_dim=STATE_DIM,
            physics_func=physics,
            neural_func=neural,
            correction_weight=0.7,
        )
        t = torch.tensor(0.0)
        z = torch.randn(BATCH_SIZE, STATE_DIM)
        u = torch.randn(BATCH_SIZE, INPUT_DIM)

        with torch.no_grad():
            physics_out = physics(t, z, u)
            neural_out = neural(t, z, u)
            hybrid_out = hybrid(t, z, u)

        expected = physics_out + 0.7 * neural_out
        torch.testing.assert_close(hybrid_out, expected, atol=1e-5, rtol=1e-5)

    def test_hybrid_inherits_input_dim_from_neural(self, mlp_func):
        """HybridODEFunc.input_dim should come from the neural_func."""
        physics = MLPODEFunc(state_dim=STATE_DIM, hidden_dim=16, num_layers=2)
        hybrid = HybridODEFunc(
            state_dim=STATE_DIM,
            physics_func=physics,
            neural_func=mlp_func,
        )
        assert hybrid.input_dim == mlp_func.input_dim


# ===================================================================
# NeuralODE Tests
# ===================================================================


class TestNeuralODE:
    """Tests for the standard Neural ODE model."""

    def test_forward_shape(self, neural_ode):
        z0 = torch.randn(BATCH_SIZE, STATE_DIM)
        t_span = torch.linspace(0, 1, NUM_TIMES)
        with torch.no_grad():
            trajectory = neural_ode(z0, t_span)
        assert trajectory.shape == (BATCH_SIZE, NUM_TIMES, STATE_DIM)

    def test_forward_single_time_step(self, neural_ode):
        """Forward with two time points (t0, t1) should give (batch, 2, state_dim)."""
        z0 = torch.randn(BATCH_SIZE, STATE_DIM)
        t_span = torch.tensor([0.0, 0.1])
        with torch.no_grad():
            trajectory = neural_ode(z0, t_span)
        assert trajectory.shape == (BATCH_SIZE, 2, STATE_DIM)

    def test_compute_loss_returns_dict(self, neural_ode):
        preds = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        targets = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        losses = neural_ode.compute_loss(preds, targets)
        assert isinstance(losses, dict)
        assert "total" in losses
        assert "data" in losses

    def test_compute_loss_total_is_scalar(self, neural_ode):
        preds = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        targets = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        losses = neural_ode.compute_loss(preds, targets)
        assert losses["total"].ndim == 0

    def test_zero_loss_for_identical_predictions(self, neural_ode):
        targets = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        losses = neural_ode.compute_loss(targets, targets)
        assert losses["total"].item() == pytest.approx(0.0, abs=1e-7)

    def test_positive_loss_for_different_predictions(self, neural_ode):
        preds = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        targets = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        losses = neural_ode.compute_loss(preds, targets)
        assert losses["total"].item() > 0.0
        assert losses["data"].item() > 0.0

    def test_compute_loss_with_custom_weights(self, neural_ode):
        preds = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        targets = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        weights = {"data": 2.5}
        losses = neural_ode.compute_loss(preds, targets, loss_weights=weights)
        data_loss = losses["data"]
        total_loss = losses["total"]
        torch.testing.assert_close(total_loss, 2.5 * data_loss)

    def test_predict_matches_forward(self, neural_ode):
        """predict() should give same result as forward() with no_grad."""
        z0 = torch.randn(BATCH_SIZE, STATE_DIM)
        t_span = torch.linspace(0, 1, NUM_TIMES)
        with torch.no_grad():
            fwd = neural_ode(z0, t_span)
        pred = neural_ode.predict(z0, t_span)
        torch.testing.assert_close(fwd, pred)

    def test_initial_condition_preserved(self, neural_ode):
        """First time step of trajectory should match z0."""
        z0 = torch.randn(BATCH_SIZE, STATE_DIM)
        t_span = torch.linspace(0, 1, NUM_TIMES)
        with torch.no_grad():
            trajectory = neural_ode(z0, t_span)
        torch.testing.assert_close(trajectory[:, 0, :], z0, atol=1e-4, rtol=1e-3)

    def test_default_creates_mlp_func(self, neural_ode):
        """When no ode_func is supplied, NeuralODE should build an MLPODEFunc."""
        assert isinstance(neural_ode.ode_func, MLPODEFunc)

    def test_custom_ode_func(self):
        """Passing a custom ode_func should use it directly."""
        custom_func = MLPODEFunc(
            state_dim=STATE_DIM, hidden_dim=16, num_layers=2
        )
        model = NeuralODE(
            state_dim=STATE_DIM,
            ode_func=custom_func,
            solver=SOLVER,
            adjoint=ADJOINT,
        )
        assert model.ode_func is custom_func

    def test_output_dim_defaults_to_state_dim(self, neural_ode):
        assert neural_ode.output_dim == STATE_DIM

    def test_output_dim_explicit(self, neural_ode_with_output_dim):
        assert neural_ode_with_output_dim.output_dim == 5

    def test_controls_warning(self, neural_ode):
        """Controls with an ODE func lacking _constant_controls should log a
        warning but not crash."""
        z0 = torch.randn(BATCH_SIZE, STATE_DIM)
        t_span = torch.linspace(0, 1, NUM_TIMES)
        controls = torch.randn(BATCH_SIZE, NUM_TIMES, INPUT_DIM)
        with torch.no_grad():
            trajectory = neural_ode(z0, t_span, controls=controls)
        # Should still produce valid output
        assert trajectory.shape == (BATCH_SIZE, NUM_TIMES, STATE_DIM)

    def test_trajectory_values_are_finite(self, neural_ode):
        z0 = torch.randn(BATCH_SIZE, STATE_DIM)
        t_span = torch.linspace(0, 1, NUM_TIMES)
        with torch.no_grad():
            trajectory = neural_ode(z0, t_span)
        assert torch.all(torch.isfinite(trajectory))

    def test_gradient_flows_through_forward(self):
        """Gradients should propagate back through the ODE solve (non-adjoint)."""
        model = NeuralODE(
            state_dim=STATE_DIM,
            solver=SOLVER,
            adjoint=ADJOINT,
            hidden_dim=HIDDEN_DIM,
            num_layers=2,
        )
        z0 = torch.randn(BATCH_SIZE, STATE_DIM, requires_grad=True)
        t_span = torch.linspace(0, 1, NUM_TIMES)
        trajectory = model(z0, t_span)
        loss = trajectory.sum()
        loss.backward()
        assert z0.grad is not None


# ===================================================================
# AugmentedNeuralODE Tests
# ===================================================================


class TestAugmentedNeuralODE:
    """Tests for the Augmented Neural ODE model."""

    def test_forward_shape(self, augmented_ode):
        """Output should have physical state_dim, not augmented."""
        z0 = torch.randn(BATCH_SIZE, STATE_DIM)
        t_span = torch.linspace(0, 1, NUM_TIMES)
        with torch.no_grad():
            trajectory = augmented_ode(z0, t_span)
        assert trajectory.shape == (BATCH_SIZE, NUM_TIMES, STATE_DIM)

    def test_augment_state_shape(self, augmented_ode):
        """augment_state should produce (batch, full_dim)."""
        z = torch.randn(BATCH_SIZE, STATE_DIM)
        z_full = augmented_ode.augment_state(z)
        assert z_full.shape == (BATCH_SIZE, STATE_DIM + AUGMENT_DIM)

    def test_augment_state_preserves_physical(self, augmented_ode):
        """Physical dimensions should be preserved after augmentation."""
        z = torch.randn(BATCH_SIZE, STATE_DIM)
        z_full = augmented_ode.augment_state(z)
        torch.testing.assert_close(z_full[:, :STATE_DIM], z)

    def test_augment_state_zeros_extra_dims(self, augmented_ode):
        """Augmented dims should be zero-initialized."""
        z = torch.randn(BATCH_SIZE, STATE_DIM)
        z_full = augmented_ode.augment_state(z)
        assert torch.all(z_full[:, STATE_DIM:] == 0)

    def test_extract_physical_shape(self, augmented_ode):
        z_full = torch.randn(BATCH_SIZE, STATE_DIM + AUGMENT_DIM)
        z_phys = augmented_ode.extract_physical(z_full)
        assert z_phys.shape == (BATCH_SIZE, STATE_DIM)

    def test_extract_physical_3d(self, augmented_ode):
        """extract_physical should work on (batch, time, full_dim) tensors."""
        z_full = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM + AUGMENT_DIM)
        z_phys = augmented_ode.extract_physical(z_full)
        assert z_phys.shape == (BATCH_SIZE, NUM_TIMES, STATE_DIM)

    def test_augment_extract_roundtrip(self, augmented_ode):
        """extract_physical(augment_state(z)) should recover z exactly."""
        z = torch.randn(BATCH_SIZE, STATE_DIM)
        z_roundtrip = augmented_ode.extract_physical(augmented_ode.augment_state(z))
        torch.testing.assert_close(z_roundtrip, z)

    def test_full_dim_attribute(self, augmented_ode):
        assert augmented_ode.full_dim == STATE_DIM + AUGMENT_DIM
        assert augmented_ode.augment_dim == AUGMENT_DIM

    def test_compute_loss_returns_dict(self, augmented_ode):
        preds = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        targets = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        losses = augmented_ode.compute_loss(preds, targets)
        assert isinstance(losses, dict)
        assert "total" in losses
        assert "data" in losses

    def test_compute_loss_zero_for_identical(self, augmented_ode):
        targets = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        losses = augmented_ode.compute_loss(targets, targets)
        assert losses["data"].item() == pytest.approx(0.0, abs=1e-7)

    def test_get_augmented_trajectory_shape(self, augmented_ode):
        """get_augmented_trajectory should return full_dim trajectory."""
        z0 = torch.randn(BATCH_SIZE, STATE_DIM)
        t_span = torch.linspace(0, 1, NUM_TIMES)
        with torch.no_grad():
            traj_full = augmented_ode.get_augmented_trajectory(z0, t_span)
        assert traj_full.shape == (BATCH_SIZE, NUM_TIMES, STATE_DIM + AUGMENT_DIM)

    def test_forward_and_augmented_trajectory_consistent(self, augmented_ode):
        """Physical part of get_augmented_trajectory should match forward output."""
        z0 = torch.randn(BATCH_SIZE, STATE_DIM)
        t_span = torch.linspace(0, 1, NUM_TIMES)
        with torch.no_grad():
            traj_physical = augmented_ode(z0, t_span)
            traj_full = augmented_ode.get_augmented_trajectory(z0, t_span)
        extracted = augmented_ode.extract_physical(traj_full)
        torch.testing.assert_close(traj_physical, extracted, atol=1e-5, rtol=1e-5)

    def test_default_creates_mlp_in_augmented_space(self, augmented_ode):
        """Default ode_func should have state_dim = full_dim."""
        assert isinstance(augmented_ode.ode_func, MLPODEFunc)
        assert augmented_ode.ode_func.state_dim == STATE_DIM + AUGMENT_DIM

    def test_trajectory_values_are_finite(self, augmented_ode):
        z0 = torch.randn(BATCH_SIZE, STATE_DIM)
        t_span = torch.linspace(0, 1, NUM_TIMES)
        with torch.no_grad():
            trajectory = augmented_ode(z0, t_span)
        assert torch.all(torch.isfinite(trajectory))


# ===================================================================
# Encoder Tests
# ===================================================================


class TestEncoder:
    """Tests for the Encoder network."""

    def test_gru_encoder_shape(self):
        enc = Encoder(
            input_dim=STATE_DIM,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            encoder_type="gru",
        )
        x = torch.randn(BATCH_SIZE, SEQ_LEN, STATE_DIM)
        mu, logvar = enc(x)
        assert mu.shape == (BATCH_SIZE, LATENT_DIM)
        assert logvar.shape == (BATCH_SIZE, LATENT_DIM)

    def test_lstm_encoder_shape(self):
        enc = Encoder(
            input_dim=STATE_DIM,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            encoder_type="lstm",
        )
        x = torch.randn(BATCH_SIZE, SEQ_LEN, STATE_DIM)
        mu, logvar = enc(x)
        assert mu.shape == (BATCH_SIZE, LATENT_DIM)
        assert logvar.shape == (BATCH_SIZE, LATENT_DIM)

    def test_mlp_encoder_shape(self):
        enc = Encoder(
            input_dim=STATE_DIM,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            encoder_type="mlp",
        )
        x = torch.randn(BATCH_SIZE, STATE_DIM)
        mu, logvar = enc(x)
        assert mu.shape == (BATCH_SIZE, LATENT_DIM)
        assert logvar.shape == (BATCH_SIZE, LATENT_DIM)

    def test_unknown_encoder_type_raises(self):
        with pytest.raises(ValueError, match="Unknown encoder_type"):
            Encoder(
                input_dim=STATE_DIM,
                latent_dim=LATENT_DIM,
                encoder_type="transformer",
            )

    def test_encoder_output_is_finite(self):
        enc = Encoder(
            input_dim=STATE_DIM,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            encoder_type="gru",
        )
        x = torch.randn(BATCH_SIZE, SEQ_LEN, STATE_DIM)
        mu, logvar = enc(x)
        assert torch.all(torch.isfinite(mu))
        assert torch.all(torch.isfinite(logvar))


# ===================================================================
# Decoder Tests
# ===================================================================


class TestDecoder:
    """Tests for the Decoder network."""

    def test_decoder_2d_shape(self):
        dec = Decoder(
            latent_dim=LATENT_DIM,
            output_dim=STATE_DIM,
            hidden_dim=HIDDEN_DIM,
        )
        z = torch.randn(BATCH_SIZE, LATENT_DIM)
        out = dec(z)
        assert out.shape == (BATCH_SIZE, STATE_DIM)

    def test_decoder_3d_shape(self):
        """Decoder should handle (batch, time, latent_dim) inputs."""
        dec = Decoder(
            latent_dim=LATENT_DIM,
            output_dim=STATE_DIM,
            hidden_dim=HIDDEN_DIM,
        )
        z = torch.randn(BATCH_SIZE, NUM_TIMES, LATENT_DIM)
        out = dec(z)
        assert out.shape == (BATCH_SIZE, NUM_TIMES, STATE_DIM)

    def test_decoder_output_is_finite(self):
        dec = Decoder(
            latent_dim=LATENT_DIM,
            output_dim=STATE_DIM,
            hidden_dim=HIDDEN_DIM,
        )
        z = torch.randn(BATCH_SIZE, LATENT_DIM)
        out = dec(z)
        assert torch.all(torch.isfinite(out))


# ===================================================================
# LatentNeuralODE Tests
# ===================================================================


class TestLatentNeuralODE:
    """Tests for the Latent Neural ODE model."""

    def test_forward_shape_gru(self, latent_ode_gru):
        """GRU encoder path: z0 is (batch, seq_len, state_dim)."""
        z0 = torch.randn(BATCH_SIZE, SEQ_LEN, STATE_DIM)
        t_span = torch.linspace(0, 1, NUM_TIMES)
        with torch.no_grad():
            trajectory = latent_ode_gru(z0, t_span)
        assert trajectory.shape == (BATCH_SIZE, NUM_TIMES, STATE_DIM)

    def test_forward_shape_mlp(self, latent_ode_mlp):
        """MLP encoder path: z0 is (batch, state_dim)."""
        z0 = torch.randn(BATCH_SIZE, STATE_DIM)
        t_span = torch.linspace(0, 1, NUM_TIMES)
        with torch.no_grad():
            trajectory = latent_ode_mlp(z0, t_span)
        assert trajectory.shape == (BATCH_SIZE, NUM_TIMES, STATE_DIM)

    def test_encode_shape(self, latent_ode_gru):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, STATE_DIM)
        mu, logvar = latent_ode_gru.encode(x)
        assert mu.shape == (BATCH_SIZE, LATENT_DIM)
        assert logvar.shape == (BATCH_SIZE, LATENT_DIM)

    def test_reparameterize_shape(self, latent_ode_gru):
        mu = torch.randn(BATCH_SIZE, LATENT_DIM)
        logvar = torch.randn(BATCH_SIZE, LATENT_DIM)
        z = latent_ode_gru.reparameterize(mu, logvar)
        assert z.shape == (BATCH_SIZE, LATENT_DIM)

    def test_reparameterize_with_zero_variance(self, latent_ode_gru):
        """When logvar = -inf (std = 0), reparameterize should return mu."""
        mu = torch.randn(BATCH_SIZE, LATENT_DIM)
        logvar = torch.full((BATCH_SIZE, LATENT_DIM), -100.0)
        z = latent_ode_gru.reparameterize(mu, logvar)
        torch.testing.assert_close(z, mu, atol=1e-5, rtol=1e-5)

    def test_decode_shape(self, latent_ode_gru):
        z = torch.randn(BATCH_SIZE, LATENT_DIM)
        out = latent_ode_gru.decode(z)
        assert out.shape == (BATCH_SIZE, STATE_DIM)

    def test_decode_3d_shape(self, latent_ode_gru):
        z = torch.randn(BATCH_SIZE, NUM_TIMES, LATENT_DIM)
        out = latent_ode_gru.decode(z)
        assert out.shape == (BATCH_SIZE, NUM_TIMES, STATE_DIM)

    def test_compute_loss_returns_dict(self, latent_ode_gru):
        preds = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        targets = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        losses = latent_ode_gru.compute_loss(preds, targets)
        assert isinstance(losses, dict)
        assert "total" in losses
        assert "reconstruction" in losses
        assert "kl" in losses

    def test_compute_loss_total_is_scalar(self, latent_ode_gru):
        preds = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        targets = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        losses = latent_ode_gru.compute_loss(preds, targets)
        assert losses["total"].ndim == 0
        assert losses["reconstruction"].ndim == 0
        assert losses["kl"].ndim == 0

    def test_compute_loss_values_finite(self, latent_ode_gru):
        preds = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        targets = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        losses = latent_ode_gru.compute_loss(preds, targets)
        for key, val in losses.items():
            assert torch.isfinite(val), f"Loss '{key}' is not finite"

    def test_compute_loss_reconstruction_zero_for_identical(self, latent_ode_gru):
        targets = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        losses = latent_ode_gru.compute_loss(targets, targets)
        assert losses["reconstruction"].item() == pytest.approx(0.0, abs=1e-7)

    def test_compute_loss_custom_weights(self, latent_ode_gru):
        preds = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        targets = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        weights = {"reconstruction": 2.0, "kl": 0.5}
        losses = latent_ode_gru.compute_loss(preds, targets, loss_weights=weights)
        # Verify total = 2.0 * reconstruction + 0.5 * kl
        expected_total = 2.0 * losses["reconstruction"] + 0.5 * losses["kl"]
        torch.testing.assert_close(losses["total"], expected_total)

    def test_latent_dim_attribute(self, latent_ode_gru):
        assert latent_ode_gru.latent_dim == LATENT_DIM

    def test_output_dim_defaults_to_state_dim(self, latent_ode_gru):
        assert latent_ode_gru.output_dim == STATE_DIM

    def test_default_creates_mlp_ode_func_in_latent_space(self, latent_ode_gru):
        """ODE func operates in latent space, so state_dim = latent_dim."""
        assert isinstance(latent_ode_gru.ode_func, MLPODEFunc)
        assert latent_ode_gru.ode_func.state_dim == LATENT_DIM

    def test_trajectory_values_are_finite(self, latent_ode_mlp):
        z0 = torch.randn(BATCH_SIZE, STATE_DIM)
        t_span = torch.linspace(0, 1, NUM_TIMES)
        with torch.no_grad():
            trajectory = latent_ode_mlp(z0, t_span)
        assert torch.all(torch.isfinite(trajectory))

    def test_gradient_flows_through_full_pipeline(self):
        """Gradients should propagate through encode -> ODE -> decode."""
        model = LatentNeuralODE(
            state_dim=STATE_DIM,
            latent_dim=LATENT_DIM,
            encoder_type="mlp",
            solver=SOLVER,
            adjoint=ADJOINT,
            hidden_dim=HIDDEN_DIM,
            num_layers=2,
        )
        z0 = torch.randn(BATCH_SIZE, STATE_DIM, requires_grad=True)
        t_span = torch.linspace(0, 1, NUM_TIMES)
        trajectory = model(z0, t_span)
        loss = trajectory.sum()
        loss.backward()
        assert z0.grad is not None

    def test_lstm_encoder_variant(self):
        """LatentNeuralODE should work with LSTM encoder."""
        model = LatentNeuralODE(
            state_dim=STATE_DIM,
            latent_dim=LATENT_DIM,
            encoder_type="lstm",
            solver=SOLVER,
            adjoint=ADJOINT,
            hidden_dim=HIDDEN_DIM,
            num_layers=2,
        )
        z0 = torch.randn(BATCH_SIZE, SEQ_LEN, STATE_DIM)
        t_span = torch.linspace(0, 1, NUM_TIMES)
        with torch.no_grad():
            trajectory = model(z0, t_span)
        assert trajectory.shape == (BATCH_SIZE, NUM_TIMES, STATE_DIM)


# ===================================================================
# Integration / Cross-cutting Tests
# ===================================================================


class TestIntegration:
    """Cross-cutting tests that combine multiple components."""

    def test_neural_ode_train_step(self, neural_ode):
        """Full train_step should run without error and return float losses."""
        z0 = torch.randn(BATCH_SIZE, STATE_DIM)
        t_span = torch.linspace(0, 1, NUM_TIMES)
        targets = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        optimizer = torch.optim.Adam(neural_ode.parameters(), lr=1e-3)

        batch = {"z0": z0, "t_span": t_span, "targets": targets}
        losses = neural_ode.train_step(batch, optimizer)

        assert isinstance(losses, dict)
        assert isinstance(losses["total"], float)
        assert losses["total"] > 0.0

    def test_augmented_ode_train_step(self, augmented_ode):
        """AugmentedNeuralODE train_step should work end-to-end."""
        z0 = torch.randn(BATCH_SIZE, STATE_DIM)
        t_span = torch.linspace(0, 1, NUM_TIMES)
        targets = torch.randn(BATCH_SIZE, NUM_TIMES, STATE_DIM)
        optimizer = torch.optim.Adam(augmented_ode.parameters(), lr=1e-3)

        batch = {"z0": z0, "t_span": t_span, "targets": targets}
        losses = augmented_ode.train_step(batch, optimizer)

        assert isinstance(losses["total"], float)

    def test_neural_ode_with_hybrid_func(self):
        """NeuralODE should accept a HybridODEFunc as its ode_func."""
        physics = MLPODEFunc(state_dim=STATE_DIM, hidden_dim=16, num_layers=2)
        neural = MLPODEFunc(state_dim=STATE_DIM, hidden_dim=16, num_layers=2)
        hybrid = HybridODEFunc(
            state_dim=STATE_DIM,
            physics_func=physics,
            neural_func=neural,
            correction_weight=0.5,
        )
        model = NeuralODE(
            state_dim=STATE_DIM,
            ode_func=hybrid,
            solver=SOLVER,
            adjoint=ADJOINT,
        )
        z0 = torch.randn(BATCH_SIZE, STATE_DIM)
        t_span = torch.linspace(0, 1, NUM_TIMES)
        with torch.no_grad():
            trajectory = model(z0, t_span)
        assert trajectory.shape == (BATCH_SIZE, NUM_TIMES, STATE_DIM)

    def test_augmented_ode_with_custom_func(self):
        """AugmentedNeuralODE with a pre-built ode_func in augmented space."""
        full_dim = STATE_DIM + AUGMENT_DIM
        custom_func = MLPODEFunc(
            state_dim=full_dim, hidden_dim=16, num_layers=2
        )
        model = AugmentedNeuralODE(
            state_dim=STATE_DIM,
            augment_dim=AUGMENT_DIM,
            ode_func=custom_func,
            solver=SOLVER,
            adjoint=ADJOINT,
        )
        z0 = torch.randn(BATCH_SIZE, STATE_DIM)
        t_span = torch.linspace(0, 1, NUM_TIMES)
        with torch.no_grad():
            trajectory = model(z0, t_span)
        assert trajectory.shape == (BATCH_SIZE, NUM_TIMES, STATE_DIM)

    def test_different_batch_sizes(self, neural_ode):
        """Model should handle various batch sizes."""
        for bs in [1, 2, 16]:
            z0 = torch.randn(bs, STATE_DIM)
            t_span = torch.linspace(0, 1, NUM_TIMES)
            with torch.no_grad():
                trajectory = neural_ode(z0, t_span)
            assert trajectory.shape == (bs, NUM_TIMES, STATE_DIM)

    def test_model_has_trainable_params(self, neural_ode):
        """Sanity check: the model should have learnable parameters."""
        num_params = sum(p.numel() for p in neural_ode.parameters() if p.requires_grad)
        assert num_params > 0


# ===================================================================
# Save / Load Tests
# ===================================================================


class TestSaveLoad:
    """Tests for AbstractNeuralDE save/load checkpoint methods."""

    def test_save_creates_checkpoint(self, neural_ode, tmp_path):
        path = tmp_path / "model.pt"
        neural_ode.save(path)
        assert path.exists()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "state_dim" in checkpoint
        assert "input_dim" in checkpoint
        assert "output_dim" in checkpoint
        assert checkpoint["state_dim"] == STATE_DIM

    def test_load_restores_model(self, neural_ode, tmp_path):
        path = tmp_path / "model.pt"
        neural_ode.save(path)

        loaded = NeuralODE.load(
            path,
            solver=SOLVER,
            adjoint=ADJOINT,
            hidden_dim=HIDDEN_DIM,
            num_layers=2,
        )
        assert loaded.state_dim == neural_ode.state_dim

        z0 = torch.randn(BATCH_SIZE, STATE_DIM)
        t_span = torch.linspace(0, 1, NUM_TIMES)
        pred_orig = neural_ode.predict(z0, t_span)
        pred_loaded = loaded.predict(z0, t_span)
        torch.testing.assert_close(pred_orig, pred_loaded)

    def test_predict_runs_in_eval_mode(self, neural_ode):
        neural_ode.train()
        z0 = torch.randn(BATCH_SIZE, STATE_DIM)
        t_span = torch.linspace(0, 1, NUM_TIMES)
        neural_ode.predict(z0, t_span)
        assert not neural_ode.training

    def test_save_creates_parent_directories(self, neural_ode, tmp_path):
        path = tmp_path / "sub" / "dir" / "model.pt"
        neural_ode.save(path)
        assert path.exists()


# ===================================================================
# AbstractODEFunc Tests — line 53
# ===================================================================


class TestAbstractODEFunc:
    """Tests for abstract base class coverage."""

    def test_abstract_forward_raises_not_implemented(self):
        """Calling forward on a concrete subclass that delegates to super() raises NotImplementedError."""
        from reactor_twin.core.ode_func import AbstractODEFunc

        # Create a minimal concrete subclass that just calls super().forward()
        class DummyODEFunc(AbstractODEFunc):
            def forward(self, t, z, u=None):
                return super().forward(t, z, u)

        func = DummyODEFunc(state_dim=STATE_DIM)
        t = torch.tensor(0.0)
        z = torch.randn(BATCH_SIZE, STATE_DIM)
        with pytest.raises(NotImplementedError, match="Subclasses must implement forward"):
            func(t, z)


# ===================================================================
# MLPODEFunc — unbatched u handling (line 125)
# ===================================================================


class TestMLPODEFuncUnbatched:
    """Tests for unbatched input paths in MLPODEFunc."""

    def test_unbatched_z_with_u(self):
        """When z is 1D (unbatched) and u is provided, both get unsqueezed (line 125)."""
        func = MLPODEFunc(
            state_dim=STATE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=2,
            input_dim=INPUT_DIM,
        )
        t = torch.tensor(0.0)
        z = torch.randn(STATE_DIM)  # unbatched: 1D
        u = torch.randn(INPUT_DIM)  # unbatched: 1D
        with torch.no_grad():
            dzdt = func(t, z, u=u)
        # Output should be squeezed back to 1D
        assert dzdt.shape == (STATE_DIM,)
        assert torch.all(torch.isfinite(dzdt))


# ===================================================================
# ResNetODEFunc — activation selection (lines 179-184)
# ===================================================================


class TestResNetODEFuncActivations:
    """Tests for ResNetODEFunc activation function selection."""

    def test_resnet_activation_tanh(self):
        """ResNetODEFunc with activation='tanh' (line 179-180)."""
        func = ResNetODEFunc(
            state_dim=STATE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=2,
            activation="tanh",
        )
        t = torch.tensor(0.0)
        z = torch.randn(BATCH_SIZE, STATE_DIM)
        with torch.no_grad():
            dzdt = func(t, z)
        assert dzdt.shape == (BATCH_SIZE, STATE_DIM)
        assert torch.all(torch.isfinite(dzdt))

    def test_resnet_activation_relu(self):
        """ResNetODEFunc with activation='relu' (lines 181-182)."""
        func = ResNetODEFunc(
            state_dim=STATE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=2,
            activation="relu",
        )
        t = torch.tensor(0.0)
        z = torch.randn(BATCH_SIZE, STATE_DIM)
        with torch.no_grad():
            dzdt = func(t, z)
        assert dzdt.shape == (BATCH_SIZE, STATE_DIM)
        assert torch.all(torch.isfinite(dzdt))

    def test_resnet_unknown_activation_raises(self):
        """ResNetODEFunc with unknown activation raises ValueError (lines 183-184)."""
        with pytest.raises(ValueError, match="Unknown activation"):
            ResNetODEFunc(
                state_dim=STATE_DIM,
                hidden_dim=HIDDEN_DIM,
                num_layers=2,
                activation="gelu",
            )


# ===================================================================
# ResNetODEFunc — batch time and control input (lines 208, 212)
# ===================================================================


class TestResNetODEFuncForward:
    """Tests for ResNetODEFunc forward with batch time and control inputs."""

    def test_resnet_forward_with_batch_time(self):
        """ResNetODEFunc forward with batch t (line 208)."""
        func = ResNetODEFunc(
            state_dim=STATE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=2,
        )
        t = torch.randn(BATCH_SIZE)  # batched time
        z = torch.randn(BATCH_SIZE, STATE_DIM)
        with torch.no_grad():
            dzdt = func(t, z)
        assert dzdt.shape == (BATCH_SIZE, STATE_DIM)

    def test_resnet_forward_with_control_input(self):
        """ResNetODEFunc forward with u (line 212)."""
        func = ResNetODEFunc(
            state_dim=STATE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=2,
            input_dim=INPUT_DIM,
        )
        t = torch.tensor(0.0)
        z = torch.randn(BATCH_SIZE, STATE_DIM)
        u = torch.randn(BATCH_SIZE, INPUT_DIM)
        with torch.no_grad():
            dzdt = func(t, z, u=u)
        assert dzdt.shape == (BATCH_SIZE, STATE_DIM)


# ===================================================================
# PortHamiltonianODEFunc — with control input (lines 297, 341)
# ===================================================================


class TestPortHamiltonianWithControls:
    """Tests for PortHamiltonianODEFunc with input_dim > 0."""

    def test_port_hamiltonian_with_input_dim(self):
        """PortHamiltonianODEFunc with input_dim>0 creates B parameter (line 297)."""
        func = PortHamiltonianODEFunc(
            state_dim=STATE_DIM,
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
        )
        # B should be a Parameter (not a buffer)
        assert isinstance(func.B, nn.Parameter)
        assert func.B.shape == (STATE_DIM, INPUT_DIM)

    def test_port_hamiltonian_forward_with_control(self):
        """PortHamiltonianODEFunc forward with u adds B @ u term (line 341)."""
        func = PortHamiltonianODEFunc(
            state_dim=STATE_DIM,
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
        )
        t = torch.tensor(0.0)
        z = torch.randn(BATCH_SIZE, STATE_DIM)
        u = torch.randn(BATCH_SIZE, INPUT_DIM)

        # Forward with control
        dzdt_with_u = func(t, z, u=u)
        assert dzdt_with_u.shape == (BATCH_SIZE, STATE_DIM)
        assert torch.all(torch.isfinite(dzdt_with_u))

        # Forward without control
        dzdt_no_u = func(t, z)
        assert dzdt_no_u.shape == (BATCH_SIZE, STATE_DIM)

        # The outputs should differ (B @ u contributes)
        # (unless u happens to be zero, which is extremely unlikely with random data)
        assert not torch.allclose(dzdt_with_u, dzdt_no_u, atol=1e-6)

    def test_port_hamiltonian_gradient_computation(self):
        """PortHamiltonianODEFunc computes Hamiltonian gradient (line 297 area)."""
        func = PortHamiltonianODEFunc(
            state_dim=STATE_DIM,
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
        )
        t = torch.tensor(0.0)
        z = torch.randn(BATCH_SIZE, STATE_DIM)
        u = torch.randn(BATCH_SIZE, INPUT_DIM)

        # Should produce finite, non-zero output
        dzdt = func(t, z, u=u)
        assert torch.all(torch.isfinite(dzdt))
        # Check that gradients can flow
        loss = dzdt.sum()
        loss.backward()
        assert func.B.grad is not None


# ===========================================================================
# Import guard tests for Neural CDE and Neural SDE
# ===========================================================================


class TestNeuralCDEImportGuard:
    """Test that NeuralCDE raises ImportError when torchcde is unavailable."""

    def test_import_error_when_torchcde_unavailable(self):
        """NeuralCDE __init__ raises ImportError when TORCHCDE_AVAILABLE is False."""
        import reactor_twin.core.neural_cde as cde_module

        original = cde_module.TORCHCDE_AVAILABLE
        try:
            cde_module.TORCHCDE_AVAILABLE = False
            with pytest.raises(ImportError, match="torchcde is required"):
                cde_module.NeuralCDE(state_dim=4, input_dim=3)
        finally:
            cde_module.TORCHCDE_AVAILABLE = original

    def test_unknown_interpolation_raises(self):
        """NeuralCDE forward raises ValueError for unknown interpolation."""
        pytest.importorskip("torchcde")
        from reactor_twin.core.neural_cde import NeuralCDE

        model = NeuralCDE(
            state_dim=4,
            input_dim=3,
            interpolation="linear",
            solver="euler",
            adjoint=False,
        )
        # Override interpolation to trigger unknown branch
        model.interpolation = "unknown_method"
        z0 = torch.randn(2, 3)
        t_span = torch.linspace(0, 1, 5)
        controls = torch.randn(2, 5, 3)
        with pytest.raises(ValueError, match="Unknown interpolation"):
            model(z0, t_span, controls=controls)

    def test_compute_loss_all_nan_targets(self):
        """When all targets are NaN, mask.any() is False and fallback loss is used."""
        pytest.importorskip("torchcde")
        from reactor_twin.core.neural_cde import NeuralCDE

        model = NeuralCDE(
            state_dim=4,
            input_dim=3,
            interpolation="linear",
            solver="euler",
            adjoint=False,
        )
        preds = torch.randn(2, 5, 3)
        targets = torch.full((2, 5, 3), float("nan"))
        losses = model.compute_loss(preds, targets)
        # Loss should still return a value (NaN-based MSE)
        assert "total" in losses
        assert "data" in losses


class TestNeuralSDEImportGuard:
    """Test that NeuralSDE raises ImportError when torchsde is unavailable."""

    def test_import_error_when_torchsde_unavailable(self):
        """NeuralSDE __init__ raises ImportError when TORCHSDE_AVAILABLE is False."""
        import reactor_twin.core.neural_sde as sde_module

        original = sde_module.TORCHSDE_AVAILABLE
        try:
            sde_module.TORCHSDE_AVAILABLE = False
            with pytest.raises(ImportError, match="torchsde is required"):
                sde_module.NeuralSDE(state_dim=3)
        finally:
            sde_module.TORCHSDE_AVAILABLE = original

    def test_sde_func_unsupported_noise_type_auto_creation(self):
        """SDEFunc raises ValueError for unsupported noise_type auto-creation."""
        from reactor_twin.core.neural_sde import SDEFunc

        drift = MLPODEFunc(state_dim=3, hidden_dim=16, num_layers=2)
        with pytest.raises(ValueError, match="Unsupported noise_type"):
            SDEFunc(drift, noise_type="general")

    def test_sde_func_general_with_custom_diffusion(self):
        """SDEFunc with noise_type='general' and custom diffusion function."""
        from reactor_twin.core.neural_sde import SDEFunc

        drift = MLPODEFunc(state_dim=3, hidden_dim=16, num_layers=2)

        class CustomDiffusion(nn.Module):
            def forward(self, t, z):
                # Return (batch, state_dim, noise_dim)
                batch = z.shape[0]
                return torch.ones(batch, 3, 2) * 0.1

        custom_diff = CustomDiffusion()
        sde = SDEFunc(drift, diffusion_func=custom_diff, noise_type="general")
        z = torch.randn(4, 3)
        t = torch.tensor(0.0)
        out = sde.g(t, z)
        assert out.shape == (4, 3, 2)
