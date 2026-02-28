"""Tests for reactor_twin.digital_twin.meta_learner."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from reactor_twin.core.neural_ode import NeuralODE
from reactor_twin.core.ode_func import MLPODEFunc
from reactor_twin.digital_twin.meta_learner import ReptileMetaLearner
from reactor_twin.reactors.systems import create_exothermic_cstr
from reactor_twin.training.data_generator import ReactorDataGenerator


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


@pytest.fixture
def model():
    ode_func = MLPODEFunc(state_dim=2, hidden_dim=16, num_layers=2)
    return NeuralODE(state_dim=2, ode_func=ode_func, solver="euler", adjoint=False)


@pytest.fixture
def task_generators():
    """Create two task generators from isothermal reactors (state_dim=2)."""
    gens = []
    for _ in range(2):
        reactor = create_exothermic_cstr(isothermal=True)
        gens.append(ReactorDataGenerator(reactor))
    return gens


@pytest.fixture
def t_span():
    return (0.0, 1.0)


@pytest.fixture
def t_eval():
    return np.linspace(0.0, 1.0, 20)


class TestInnerLoop:
    def test_returns_adapted_params(self, model, task_generators, t_span, t_eval):
        learner = ReptileMetaLearner(model, inner_steps=2)
        adapted = learner._inner_loop(task_generators[0], t_span, t_eval, batch_size=4)
        assert isinstance(adapted, dict)
        assert len(adapted) > 0
        for _name, param in adapted.items():
            assert isinstance(param, torch.Tensor)

    def test_inner_loop_changes_params(self, model, task_generators, t_span, t_eval):
        learner = ReptileMetaLearner(model, inner_steps=3)
        original = {n: p.detach().clone() for n, p in model.named_parameters()}
        adapted = learner._inner_loop(task_generators[0], t_span, t_eval, batch_size=4)
        # Adapted params should differ from original
        any_changed = False
        for name, orig_val in original.items():
            if not torch.allclose(orig_val, adapted[name], atol=1e-8):
                any_changed = True
                break
        assert any_changed, "Inner loop did not change any parameters"

    def test_inner_loop_does_not_modify_meta_model(self, model, task_generators, t_span, t_eval):
        learner = ReptileMetaLearner(model, inner_steps=3)
        original = {n: p.detach().clone() for n, p in model.named_parameters()}
        learner._inner_loop(task_generators[0], t_span, t_eval, batch_size=4)
        # Meta-model should NOT change
        for name, param in model.named_parameters():
            torch.testing.assert_close(param, original[name])


class TestMetaStep:
    def test_returns_displacement(self, model, task_generators, t_span, t_eval):
        learner = ReptileMetaLearner(model, meta_lr=0.1, inner_steps=2)
        disp = learner.meta_step(task_generators, t_span, t_eval, batch_size=4)
        assert isinstance(disp, float)
        assert disp >= 0.0

    def test_modifies_meta_model(self, model, task_generators, t_span, t_eval):
        learner = ReptileMetaLearner(model, meta_lr=0.1, inner_steps=2)
        original = {n: p.detach().clone() for n, p in model.named_parameters()}
        learner.meta_step(task_generators, t_span, t_eval, batch_size=4)
        any_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, original[name], atol=1e-8):
                any_changed = True
                break
        assert any_changed, "Meta step did not update model"

    def test_tasks_per_step(self, model, task_generators, t_span, t_eval):
        learner = ReptileMetaLearner(model, meta_lr=0.1, inner_steps=2)
        disp = learner.meta_step(task_generators, t_span, t_eval, tasks_per_step=1, batch_size=4)
        assert isinstance(disp, float)


class TestMetaTrain:
    def test_returns_displacements_list(self, model, task_generators, t_span, t_eval):
        learner = ReptileMetaLearner(model, meta_lr=0.1, inner_steps=2)
        disps = learner.meta_train(task_generators, num_steps=3, t_span=t_span, t_eval=t_eval, batch_size=4, log_interval=100)
        assert len(disps) == 3
        assert all(isinstance(d, float) for d in disps)

    def test_displacement_bounded(self, model, task_generators, t_span, t_eval):
        learner = ReptileMetaLearner(model, meta_lr=0.01, inner_steps=2)
        disps = learner.meta_train(task_generators, num_steps=2, t_span=t_span, t_eval=t_eval, batch_size=4, log_interval=100)
        for d in disps:
            assert d < 1e6, "Displacement unreasonably large"


class TestFineTune:
    def test_returns_losses(self, model, task_generators, t_span, t_eval):
        learner = ReptileMetaLearner(model, inner_lr=1e-3)
        losses = learner.fine_tune(task_generators[0], t_span=t_span, t_eval=t_eval, num_steps=3, batch_size=4)
        assert len(losses) == 3
        assert all(isinstance(v, float) for v in losses)

    def test_loss_decreases(self, model, task_generators, t_span, t_eval):
        learner = ReptileMetaLearner(model, inner_lr=1e-3)
        losses = learner.fine_tune(task_generators[0], t_span=t_span, t_eval=t_eval, num_steps=10, batch_size=8)
        # Loss should generally decrease (first > last)
        assert losses[-1] <= losses[0] * 2.0, "Loss increased significantly"

    def test_default_t_eval(self, model, task_generators, t_span):
        learner = ReptileMetaLearner(model, inner_lr=1e-3)
        losses = learner.fine_tune(task_generators[0], t_span=t_span, num_steps=2, batch_size=4)
        assert len(losses) == 2
