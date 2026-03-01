"""End-to-end integration tests that cross module boundaries.

Each test exercises a full pipeline: reactor -> data -> model -> downstream.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from reactor_twin.core.bayesian_neural_ode import BayesianNeuralODE
from reactor_twin.core.hybrid_model import HybridNeuralODE
from reactor_twin.core.neural_ode import NeuralODE
from reactor_twin.digital_twin.fault_detector import SPCChart
from reactor_twin.digital_twin.mpc_controller import MPCController
from reactor_twin.digital_twin.state_estimator import EKFStateEstimator
from reactor_twin.physics.constraints import ConstraintPipeline
from reactor_twin.physics.mass_balance import MassBalanceConstraint
from reactor_twin.physics.positivity import PositivityConstraint
from reactor_twin.reactors.systems import create_exothermic_cstr
from reactor_twin.training.data_generator import ReactorDataGenerator
from reactor_twin.training.foundation import FoundationNeuralODE, FoundationTrainer
from reactor_twin.training.trainer import Trainer
from reactor_twin.utils.registry import NEURAL_DE_REGISTRY, REACTOR_REGISTRY

# ── helpers ──────────────────────────────────────────────────────────


def _generate_cstr_data(
    isothermal: bool = True,
    num_points: int = 20,
    t_end: float = 1.0,
) -> dict[str, np.ndarray]:
    """Generate a single trajectory from the exothermic CSTR."""
    reactor = create_exothermic_cstr(isothermal=isothermal)
    gen = ReactorDataGenerator(reactor)
    t_eval = np.linspace(0, t_end, num_points)
    return gen.generate_trajectory((0, t_end), t_eval)


# ── 1. CSTR -> Data -> NeuralODE -> train step ──────────────────────


@pytest.mark.slow
class TestCSTRNeuralODETrainStep:
    """Create exothermic CSTR, generate data, build NeuralODE, do one training step."""

    def test_forward_loss_and_gradient_step(self) -> None:
        traj = _generate_cstr_data(isothermal=True, num_points=20, t_end=1.0)
        assert traj["success"]

        y = torch.tensor(traj["y"], dtype=torch.float32)  # (T, 2)
        t = torch.tensor(traj["t"], dtype=torch.float32)  # (T,)
        z0 = y[0].unsqueeze(0)  # (1, 2)
        targets = y.unsqueeze(0)  # (1, T, 2)

        model = NeuralODE(
            state_dim=2,
            solver="rk4",
            adjoint=False,
            hidden_dim=32,
            num_layers=2,
        )

        # Forward pass
        pred = model(z0, t)  # (1, T, 2)
        assert pred.shape == targets.shape

        # Loss
        losses = model.compute_loss(pred, targets)
        loss = losses["total"]
        assert torch.isfinite(loss)

        # Backward + gradient clipping + step
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # After step the loss should still be finite
        pred2 = model(z0, t)
        loss2 = model.compute_loss(pred2, targets)["total"]
        assert torch.isfinite(loss2)


# ── 2. CSTR -> Data -> NeuralODE -> PositivityConstraint ────────────


@pytest.mark.slow
class TestCSTRWithPositivity:
    """Apply PositivityConstraint to NeuralODE predictions."""

    def test_positivity_constraint(self) -> None:
        traj = _generate_cstr_data()
        y = torch.tensor(traj["y"], dtype=torch.float32)
        t = torch.tensor(traj["t"], dtype=torch.float32)
        z0 = y[0].unsqueeze(0)

        model = NeuralODE(
            state_dim=2,
            solver="rk4",
            adjoint=False,
            hidden_dim=32,
            num_layers=2,
        )

        pred = model(z0, t)  # (1, T, 2)

        constraint = PositivityConstraint()
        z_pos, violation = constraint(pred)

        # All values must be >= 0 after hard projection
        assert z_pos.min().item() >= 0.0
        assert z_pos.shape == pred.shape


# ── 3. CSTR -> Data -> NeuralODE -> EKF ─────────────────────────────


@pytest.mark.slow
class TestCSTRNeuralODEEKF:
    """Run EKF state estimator on noisy measurements from CSTR data."""

    def test_ekf_filter(self) -> None:
        num_points = 15
        traj = _generate_cstr_data(num_points=num_points, t_end=0.5)
        y = torch.tensor(traj["y"], dtype=torch.float32)  # (T, 2)
        t = torch.tensor(traj["t"], dtype=torch.float32)

        # Add noise to create "measurements"
        noise = torch.randn_like(y) * 0.01
        measurements = y + noise  # (T, 2)

        model = NeuralODE(
            state_dim=2,
            input_dim=0,
            solver="euler",
            adjoint=False,
            hidden_dim=32,
            num_layers=2,
        )

        ekf = EKFStateEstimator(
            model=model,
            state_dim=2,
            Q=1e-4,
            R=1e-2,
            dt=0.01,
        )

        result = ekf.filter(measurements, z0=y[0], t_span=t)

        assert "states" in result
        assert "covariances" in result
        assert "innovations" in result
        assert result["states"].shape == (num_points, 2)
        assert result["covariances"].shape == (num_points, 2, 2)
        assert result["innovations"].shape == (num_points, 2)


# ── 4. CSTR -> Data -> NeuralODE -> MPC ─────────────────────────────


@pytest.mark.slow
class TestCSTRNeuralODEMPC:
    """Run MPC controller with a NeuralODE plant model."""

    def test_mpc_step(self) -> None:
        traj = _generate_cstr_data()
        y = torch.tensor(traj["y"], dtype=torch.float32)

        model = NeuralODE(
            state_dim=2,
            input_dim=1,
            solver="euler",
            adjoint=False,
            hidden_dim=32,
            num_layers=2,
        )

        controller = MPCController(
            model=model,
            horizon=5,
            dt=0.01,
            max_iter=5,
        )

        z_current = y[0]
        y_ref = y[-1]

        u_applied, info = controller.step(z_current, y_ref)

        assert u_applied.shape == (1,)  # input_dim = 1
        assert isinstance(info, dict)
        assert "controls" in info
        assert "trajectory" in info
        assert "cost" in info
        assert "converged" in info


# ── 5. CSTR -> Data -> NeuralODE -> FaultDetector (SPCChart) ────────


@pytest.mark.slow
class TestCSTRNeuralODESPCFault:
    """Create SPCChart, set baseline with normal data, detect anomalous data."""

    def test_spc_alarm_structure(self) -> None:
        traj = _generate_cstr_data(num_points=50, t_end=2.0)
        y = traj["y"]  # (T, 2) numpy

        spc = SPCChart(num_vars=2)
        # Use first 30 points as baseline (normal operation)
        spc.set_baseline(y[:30])

        # Update with an anomalous point (large deviation)
        anomalous = y[-1] + 50.0  # big shift
        result = spc.update(anomalous)

        assert "ewma_alarm" in result
        assert "cusum_alarm" in result
        assert "ewma_values" in result
        assert "cusum_pos" in result
        assert "cusum_neg" in result
        assert result["ewma_alarm"].shape == (2,)
        assert result["cusum_alarm"].shape == (2,)
        # At least one alarm should trigger for the large anomalous shift
        assert result["ewma_alarm"].any() or result["cusum_alarm"].any()


# ── 6. Multi-constraint pipeline ────────────────────────────────────


@pytest.mark.slow
class TestMultiConstraintPipeline:
    """Apply ConstraintPipeline with Positivity + MassBalance."""

    def test_pipeline_preserves_shape(self) -> None:
        model = NeuralODE(
            state_dim=2,
            solver="rk4",
            adjoint=False,
            hidden_dim=32,
            num_layers=2,
        )

        traj = _generate_cstr_data()
        y = torch.tensor(traj["y"], dtype=torch.float32)
        t = torch.tensor(traj["t"], dtype=torch.float32)
        z0 = y[0].unsqueeze(0)

        pred = model(z0, t)  # (1, T, 2)

        pipeline = ConstraintPipeline(
            [
                PositivityConstraint(),
                MassBalanceConstraint(),
            ]
        )

        z_out, violations = pipeline(pred)
        assert z_out.shape == pred.shape


# ── 7. ReactorDataGenerator integration ─────────────────────────────


@pytest.mark.slow
class TestReactorDataGeneratorIntegration:
    """Use ReactorDataGenerator.generate_dataset with isothermal CSTR."""

    def test_generate_dataset_shapes(self) -> None:
        reactor = create_exothermic_cstr(isothermal=True)
        gen = ReactorDataGenerator(reactor)
        t_eval = np.linspace(0, 1.0, 15)

        dataset = gen.generate_dataset(
            num_trajectories=4,
            t_span=(0, 1.0),
            t_eval=t_eval,
            batch_size=2,
        )

        assert len(dataset) == 2  # 4 trajectories / batch_size 2 = 2 batches

        batch = dataset[0]
        assert "z0" in batch
        assert "t_span" in batch
        assert "targets" in batch
        assert batch["z0"].shape[0] == 2  # batch_size
        assert batch["z0"].shape[1] == 2  # state_dim (C_A, C_B)
        assert batch["targets"].shape == (2, 15, 2)


# ── 8. Trainer smoke test ───────────────────────────────────────────


@pytest.mark.slow
class TestTrainerSmokeTest:
    """Create NeuralODE + Trainer, train 1 epoch on a tiny dataset."""

    def test_train_one_epoch(self) -> None:
        reactor = create_exothermic_cstr(isothermal=True)
        gen = ReactorDataGenerator(reactor)
        t_eval = np.linspace(0, 0.5, 10)

        model = NeuralODE(
            state_dim=2,
            solver="euler",
            adjoint=False,
            hidden_dim=32,
            num_layers=2,
        )

        trainer = Trainer(model=model, data_generator=gen)

        history = trainer.train(
            num_epochs=1,
            t_span=(0, 0.5),
            t_eval=t_eval,
            train_trajectories=2,
            val_trajectories=2,
            batch_size=2,
            val_interval=1,
        )

        assert "train_loss" in history
        assert len(history["train_loss"]) == 1
        assert np.isfinite(history["train_loss"][0])


# ── 9. Bayesian Neural ODE uncertainty ────────────────────────────────


@pytest.mark.slow
class TestBayesianUncertainty:
    """BayesianNeuralODE should produce non-trivial uncertainty estimates."""

    def test_bayesian_uncertainty_on_cstr(self) -> None:
        traj = _generate_cstr_data(isothermal=True, num_points=10, t_end=0.5)
        y = torch.tensor(traj["y"], dtype=torch.float32)
        t = torch.tensor(traj["t"], dtype=torch.float32)
        z0 = y[0].unsqueeze(0)

        model = BayesianNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        mean, std = model.predict_with_uncertainty(z0, t, num_samples=10)
        assert mean.shape == (1, 10, 2)
        assert std.shape == (1, 10, 2)
        assert torch.all(torch.isfinite(mean))

    def test_bayesian_loss_has_kl(self) -> None:
        model = BayesianNeuralODE(
            state_dim=2, solver="euler", hidden_dim=16, num_layers=2, beta=1.0
        )
        z0 = torch.randn(2, 2)
        t = torch.linspace(0, 0.1, 5)
        pred = model(z0, t)
        targets = torch.randn_like(pred)
        losses = model.compute_loss(pred, targets)
        assert losses["kl"].item() >= 0


# ── 10. Hybrid model + CSTR end-to-end ────────────────────────────────


@pytest.mark.slow
class TestHybridCSTREndToEnd:
    """HybridNeuralODE with CSTR physics + neural correction."""

    def test_hybrid_cstr_forward_and_loss(self) -> None:
        reactor = create_exothermic_cstr(isothermal=True)
        model = HybridNeuralODE(
            state_dim=reactor.state_dim,
            reactor=reactor,
            solver="euler",
            hidden_dim=16,
            num_layers=2,
        )
        z0 = torch.tensor(reactor.get_initial_state(), dtype=torch.float32).unsqueeze(0)
        t = torch.linspace(0, 0.02, 3)
        pred = model(z0, t)
        assert pred.shape == (1, 3, reactor.state_dim)

        targets = torch.randn_like(pred)
        losses = model.compute_loss(pred, targets)
        assert torch.isfinite(losses["total"])
        assert "physics_reg" in losses

    def test_hybrid_correction_magnitude(self) -> None:
        reactor = create_exothermic_cstr(isothermal=True)
        model = HybridNeuralODE(
            state_dim=reactor.state_dim,
            reactor=reactor,
            solver="euler",
            hidden_dim=16,
            num_layers=2,
        )
        z0 = torch.tensor(reactor.get_initial_state(), dtype=torch.float32).unsqueeze(0)
        t = torch.linspace(0, 0.02, 3)
        ratio = model.get_correction_magnitude(z0, t)
        assert torch.isfinite(ratio)


# ── 11. Foundation pretrain + fine-tune ───────────────────────────────


@pytest.mark.slow
class TestFoundationPretrainFineTune:
    """Foundation model meta-learning workflow."""

    def test_pretrain_then_fine_tune(self) -> None:
        model = FoundationNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        trainer = FoundationTrainer(model, inner_steps=2)

        # Create 2 different reactors
        generators = []
        for i in range(2):
            reactor = create_exothermic_cstr(isothermal=True)
            reactor.params["V"] = 10.0 + i * 5.0
            generators.append(ReactorDataGenerator(reactor))

        # Pretrain
        displacements = trainer.pretrain(generators, num_epochs=2, t_span=(0, 0.5), batch_size=2)
        assert len(displacements) == 2

        # Fine-tune on first reactor
        losses = trainer.fine_tune(generators[0], num_steps=3, t_span=(0, 0.5), batch_size=2)
        assert len(losses) == 3
        assert all(np.isfinite(v) for v in losses)

    def test_evaluate_transfer(self) -> None:
        model = FoundationNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        trainer = FoundationTrainer(model)
        reactor = create_exothermic_cstr(isothermal=True)
        gen = ReactorDataGenerator(reactor)
        result = trainer.evaluate_transfer([gen], t_span=(0, 0.5), batch_size=2)
        assert "mean_loss" in result
        assert np.isfinite(result["mean_loss"])


# ── 12. Sensitivity on CSTR ──────────────────────────────────────────


@pytest.mark.slow
class TestSensitivityCSTR:
    """Sensitivity analysis on CSTR parameters."""

    def test_sobol_on_cstr(self) -> None:
        pytest.importorskip("SALib")
        from reactor_twin.utils.sensitivity import SensitivityAnalyzer

        reactor = create_exothermic_cstr(isothermal=True)
        analyzer = SensitivityAnalyzer(
            reactor=reactor,
            param_names=["V", "F"],
            param_bounds=[[5.0, 20.0], [0.5, 5.0]],
        )
        t_eval = np.linspace(0, 1.0, 20)
        result = analyzer.analyze(t_span=(0, 1.0), t_eval=t_eval, method="sobol", N=32)
        assert "S1" in result
        assert len(result["S1"]) == 2


# ── 13. Registry completeness ────────────────────────────────────────


class TestRegistryCompleteness:
    """Verify all v0.3.0 entries are registered."""

    def test_reactor_registry_complete(self) -> None:
        expected = [
            "cstr",
            "batch",
            "pfr",
            "semi_batch",
            "multi_phase",
            "population_balance",
            "membrane",
            "fluidized_bed",
        ]
        for key in expected:
            assert key in REACTOR_REGISTRY, f"Missing reactor: {key}"

    def test_neural_de_registry_complete(self) -> None:
        expected = [
            "neural_ode",
            "augmented_neural_ode",
            "bayesian_neural_ode",
            "hybrid_neural_ode",
            "foundation_neural_ode",
            "latent_neural_ode",
            "neural_sde",
            "neural_cde",
        ]
        for key in expected:
            assert key in NEURAL_DE_REGISTRY, f"Missing neural DE: {key}"

    def test_version_is_1_1_0(self) -> None:
        import reactor_twin

        assert reactor_twin.__version__ == "1.1.0"


# ── 14. Membrane reactor integration ─────────────────────────────────


@pytest.mark.slow
class TestMembraneIntegration:
    """End-to-end membrane reactor simulation."""

    def test_membrane_data_generation(self) -> None:
        from reactor_twin.reactors.membrane import MembraneReactor

        params = {
            "V_ret": 10.0,
            "V_perm": 5.0,
            "F_ret": 1.0,
            "F_perm": 0.5,
            "A_membrane": 2.0,
            "Q": [0.1],
            "permeating_species_indices": [0],
            "permeation_law": "linear",
            "C_ret_feed": [1.0, 0.5],
            "T_feed": 350.0,
        }
        reactor = MembraneReactor("test", 2, params)
        gen = ReactorDataGenerator(reactor)
        t_eval = np.linspace(0, 1.0, 20)
        traj = gen.generate_trajectory((0, 1.0), t_eval)
        assert traj["success"]
        assert traj["y"].shape == (20, 3)  # 2 ret + 1 perm


# ── 15. Fluidized bed reactor integration ────────────────────────────


@pytest.mark.slow
class TestFluidizedBedIntegration:
    """End-to-end fluidized bed reactor simulation."""

    def test_fluidized_bed_data_generation(self) -> None:
        from reactor_twin.reactors.fluidized_bed import FluidizedBedReactor

        params = {
            "u_mf": 0.05,
            "u_0": 0.20,
            "epsilon_mf": 0.45,
            "d_b": 0.05,
            "H_bed": 2.0,
            "A_bed": 1.0,
            "K_be": 0.5,
            "C_feed": [1.0, 0.0],
            "T_feed": 400.0,
        }
        reactor = FluidizedBedReactor("test", 2, params)
        gen = ReactorDataGenerator(reactor)
        t_eval = np.linspace(0, 1.0, 20)
        traj = gen.generate_trajectory((0, 1.0), t_eval)
        assert traj["success"]
        assert traj["y"].shape == (20, 4)  # 2 bubble + 2 emulsion
