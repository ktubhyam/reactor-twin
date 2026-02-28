"""Penicillin Fed-Batch Fermentation Benchmark.

Models a fed-batch bioreactor for penicillin production based on the
Bajpai-Reuss kinetics model.  This is a standard benchmark for
bioprocess digital twins.

Reference:
    Bajpai, R.K. and Reuss, M. (1980), "A Mechanistic Model for
    Penicillin Production", J. Chemical Technology and Biotechnology,
    30, 332-344.

State vector: [X, S, P, V]
    X: Biomass concentration (g/L)
    S: Substrate (glucose) concentration (g/L)
    P: Penicillin concentration (g/L)
    V: Reactor volume (L)

Control input: [F_s]
    F_s: Substrate feed rate (L/h)
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)

# Kinetic parameters (Bajpai-Reuss model)
_MU_MAX = 0.11      # max specific growth rate (1/h)
_K_S = 0.006        # substrate saturation constant (g/L)
_K_I = 0.1          # substrate inhibition constant (g/L)
_K_P = 0.0001       # product inhibition on growth (L/g)
_Y_XS = 0.47        # yield coefficient X/S (g/g)
_M_S = 0.029        # maintenance coefficient (g/(g*h))
_ALPHA = 0.012      # growth-associated product formation (g/g)
_BETA = 0.00015     # non-growth-associated product formation (g/(g*h))
_K_H = 0.04         # penicillin hydrolysis constant (1/h)
_S_FEED = 400.0     # feed substrate concentration (g/L)


class PenicillinReactor:
    """Penicillin fed-batch bioreactor model.

    State: [X, S, P, V]
    Control: [F_s] (substrate feed rate, L/h)
    """

    state_dim = 4
    input_dim = 1
    state_labels = ["X (g/L)", "S (g/L)", "P (g/L)", "V (L)"]

    def __init__(self) -> None:
        pass

    def ode_rhs(
        self,
        t: float,
        y: np.ndarray,
        u: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute dy/dt."""
        X, S, P, V = y

        # Default feed rate
        F_s = 0.05 if u is None else float(u[0])

        # Specific growth rate (Monod + substrate inhibition + product inhibition)
        mu = _MU_MAX * S / ((_K_S + S) * (1 + S / _K_I)) * np.exp(-_K_P * P)

        # Dilution rate
        D = F_s / max(V, 1e-6)

        # Mass balances
        dX = mu * X - D * X
        dS = -mu * X / _Y_XS - _M_S * X + D * (_S_FEED - S)
        dP = _ALPHA * mu * X + _BETA * X - _K_H * P - D * P
        dV = F_s

        return np.array([dX, dS, dP, dV])

    def get_initial_state(self) -> np.ndarray:
        return np.array([1.5, 0.0, 0.0, 7.0])

    def get_state_labels(self) -> list[str]:
        return list(self.state_labels)


def generate_synthetic_data(
    n_trajectories: int = 50,
    t_end: float = 200.0,
    n_points: int = 400,
    noise_std: float = 0.005,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate diverse training trajectories.

    Varies the initial biomass, feed rate profiles, and initial volume
    to produce a realistic training set.

    Args:
        n_trajectories: Number of trajectories.
        t_end: Batch duration (hours).
        n_points: Time points.
        noise_std: Relative observation noise.
        seed: Random seed.

    Returns:
        Dict with ``t``, ``y0``, ``trajectories``, ``controls``.
    """
    rng = np.random.default_rng(seed)
    reactor = PenicillinReactor()
    t_eval = np.linspace(0, t_end, n_points)

    all_y0 = []
    all_traj = []
    all_controls = []

    for _ in range(n_trajectories):
        X0 = 1.0 + 1.0 * rng.random()
        S0 = 0.0 + 0.5 * rng.random()
        V0 = 5.0 + 4.0 * rng.random()
        y0 = np.array([X0, S0, 0.0, V0])

        # Time-varying feed rate: constant rate with perturbation
        F_base = 0.03 + 0.04 * rng.random()

        sol = solve_ivp(
            lambda t, y, _f=F_base: reactor.ode_rhs(t, y, np.array([_f])),
            [0, t_end],
            y0,
            t_eval=t_eval,
            method="Radau",
            rtol=1e-8,
            atol=1e-10,
        )

        if sol.success:
            traj = sol.y.T
            # Add multiplicative noise
            noise = 1.0 + noise_std * rng.standard_normal(traj.shape)
            traj = traj * noise
            traj[:, :3] = np.maximum(traj[:, :3], 0.0)
            traj[:, 3] = np.maximum(traj[:, 3], 1.0)  # volume > 0
            all_y0.append(y0)
            all_traj.append(traj)
            all_controls.append(np.array([F_base]))
        else:
            logger.warning(f"Trajectory failed: {sol.message}")

    logger.info(f"Generated {len(all_traj)}/{n_trajectories} penicillin trajectories")

    return {
        "t": t_eval,
        "y0": np.array(all_y0),
        "trajectories": np.array(all_traj),
        "controls": np.array(all_controls),
    }


def run_benchmark(
    n_train: int = 40,
    n_test: int = 10,
    hidden_dim: int = 64,
    num_epochs: int = 100,
) -> dict[str, float]:
    """Train and evaluate a Neural ODE on penicillin data.

    Args:
        n_train: Training trajectories.
        n_test: Test trajectories.
        hidden_dim: Hidden dimension.
        num_epochs: Training epochs.

    Returns:
        Dict with evaluation metrics.
    """
    import torch

    from reactor_twin.core.neural_ode import NeuralODE
    from reactor_twin.core.ode_func import MLPODEFunc
    from reactor_twin.utils.metrics import relative_rmse, rmse

    data = generate_synthetic_data(n_trajectories=n_train + n_test)

    train_traj = data["trajectories"][:n_train]
    test_traj = data["trajectories"][n_train : n_train + n_test]
    train_y0 = data["y0"][:n_train]
    test_y0 = data["y0"][n_train : n_train + n_test]
    t = data["t"]

    ode_func = MLPODEFunc(state_dim=4, hidden_dim=hidden_dim, num_layers=3)
    model = NeuralODE(ode_func, state_dim=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    t_tensor = torch.tensor(t, dtype=torch.float32)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i in range(n_train):
            z0 = torch.tensor(train_y0[i], dtype=torch.float32).unsqueeze(0)
            target = torch.tensor(train_traj[i], dtype=torch.float32).unsqueeze(0)
            pred = model(z0=z0, t_span=t_tensor)
            loss = torch.nn.functional.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss / n_train:.6f}")

    model.eval()
    all_pred = []
    all_true = []
    with torch.no_grad():
        for i in range(min(n_test, len(test_traj))):
            z0 = torch.tensor(test_y0[i], dtype=torch.float32).unsqueeze(0)
            pred = model(z0=z0, t_span=t_tensor).squeeze(0).numpy()
            all_pred.append(pred)
            all_true.append(test_traj[i])

    pred_arr = np.concatenate(all_pred, axis=0)
    true_arr = np.concatenate(all_true, axis=0)

    results = {
        "rmse": rmse(pred_arr, true_arr),
        "relative_rmse": relative_rmse(pred_arr, true_arr),
        "n_train": n_train,
        "n_test": n_test,
        "num_epochs": num_epochs,
    }

    logger.info(f"Penicillin benchmark: RMSE={results['rmse']:.4f}, "
                f"Relative RMSE={results['relative_rmse']:.2f}%")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_benchmark()
    print(f"\nResults: {results}")
