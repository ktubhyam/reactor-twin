"""Williams-Otto Reactor Benchmark.

The Williams-Otto reactor is a classic benchmark for process control
and optimisation.  It models an exothermic CSTR with three irreversible
reactions:

    A + B -> C     (k1)
    C + B -> P + E (k2)
    P + C -> G     (k3)

where P is the desired product.

Reference:
    Williams, T.J. and Otto, R.E. (1960), "A Generalized Chemical
    Processing Model for the Investigation of Computer Control",
    AIEE Transactions, 79, 458-473.

This module provides:
    1. ``WilliamsOttoReactor`` — the 6-state ODE model.
    2. ``generate_synthetic_data`` — generates realistic training data.
    3. ``run_benchmark`` — trains and evaluates a Neural ODE on the data.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)

# Kinetic parameters (Arrhenius form)
_K1_0 = 1.6599e6   # pre-exponential factor (1/min)
_K2_0 = 7.2117e8   # pre-exponential factor (1/min)
_K3_0 = 2.6745e12  # pre-exponential factor (1/min)
_E1 = 6666.7       # activation energy / R  (K)
_E2 = 8333.3
_E3 = 11111.0

# Heat of reactions (J/mol)
_DH1 = -6500.0
_DH2 = -8000.0
_DH3 = -11000.0

# Physical parameters
_RHO_CP = 500.0  # rho * Cp  (J/(L*K))


class WilliamsOttoReactor:
    """Williams-Otto CSTR reactor model.

    State vector: [C_A, C_B, C_C, C_P, C_E, T]
    Control inputs: [F_A, F_B, T_cool]  (feed rates + cooling temperature)
    """

    state_dim = 6
    input_dim = 3
    state_labels = ["C_A", "C_B", "C_C", "C_P", "C_E", "T"]

    def __init__(
        self,
        V: float = 2105.0,
        F_total: float = 20.0,
    ) -> None:
        self.V = V
        self.F_total = F_total

    def ode_rhs(
        self,
        t: float,
        y: np.ndarray,
        u: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute dy/dt for the Williams-Otto reactor."""
        C_A, C_B, C_C, C_P, C_E, T = y

        # Default controls
        if u is None:
            F_A = 10.0   # mol/min
            F_B = 10.0
            T_cool = 350.0
        else:
            F_A, F_B, T_cool = u

        # Reaction rates (Arrhenius)
        k1 = _K1_0 * np.exp(-_E1 / T)
        k2 = _K2_0 * np.exp(-_E2 / T)
        k3 = _K3_0 * np.exp(-_E3 / T)

        r1 = k1 * C_A * C_B
        r2 = k2 * C_C * C_B
        r3 = k3 * C_P * C_C

        # Dilution rate
        tau = self.V / self.F_total

        # Mass balances
        dC_A = (F_A / self.V) - (C_A / tau) - r1
        dC_B = (F_B / self.V) - (C_B / tau) - r1 - r2
        dC_C = -(C_C / tau) + r1 - r2 - r3
        dC_P = -(C_P / tau) + r2 - r3
        dC_E = -(C_E / tau) + r2
        # Energy balance
        Q_rxn = _DH1 * r1 + _DH2 * r2 + _DH3 * r3
        UA = 100.0  # heat transfer coeff * area  (J/(min*K))
        dT = Q_rxn / _RHO_CP - (T - 300.0) / tau + UA * (T_cool - T) / (_RHO_CP * self.V)

        return np.array([dC_A, dC_B, dC_C, dC_P, dC_E, dT])

    def get_initial_state(self) -> np.ndarray:
        return np.array([2.0, 2.0, 0.0, 0.0, 0.0, 350.0])

    def get_state_labels(self) -> list[str]:
        return list(self.state_labels)


def generate_synthetic_data(
    n_trajectories: int = 50,
    t_end: float = 100.0,
    n_points: int = 200,
    noise_std: float = 0.01,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate synthetic training data from the Williams-Otto model.

    Varies the initial conditions and control inputs to produce
    diverse training trajectories.

    Args:
        n_trajectories: Number of trajectories.
        t_end: End time (min).
        n_points: Time points per trajectory.
        noise_std: Observation noise standard deviation.
        seed: Random seed.

    Returns:
        Dict with keys ``t``, ``y0``, ``trajectories``, ``controls``.
    """
    rng = np.random.default_rng(seed)
    reactor = WilliamsOttoReactor()
    t_eval = np.linspace(0, t_end, n_points)

    all_y0 = []
    all_traj = []
    all_controls = []

    base_y0 = reactor.get_initial_state()

    for i in range(n_trajectories):
        # Perturb initial conditions
        y0 = base_y0 * (1.0 + 0.1 * rng.standard_normal(6))
        y0 = np.maximum(y0, 0.0)  # non-negative concentrations
        y0[5] = np.clip(y0[5], 300.0, 450.0)  # reasonable temperature

        # Varying controls
        F_A = 8.0 + 4.0 * rng.random()
        F_B = 8.0 + 4.0 * rng.random()
        T_cool = 330.0 + 40.0 * rng.random()
        u = np.array([F_A, F_B, T_cool])

        sol = solve_ivp(
            lambda t, y, _u=u: reactor.ode_rhs(t, y, _u),
            [0, t_end],
            y0,
            t_eval=t_eval,
            method="Radau",
            rtol=1e-8,
            atol=1e-10,
        )

        if sol.success:
            traj = sol.y.T + noise_std * rng.standard_normal(sol.y.T.shape)
            traj[:, :5] = np.maximum(traj[:, :5], 0.0)
            all_y0.append(y0)
            all_traj.append(traj)
            all_controls.append(u)
        else:
            logger.warning(f"Trajectory {i} failed: {sol.message}")

    logger.info(f"Generated {len(all_traj)}/{n_trajectories} trajectories")

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
    """Train and evaluate a Neural ODE on Williams-Otto data.

    Args:
        n_train: Training trajectories.
        n_test: Test trajectories.
        hidden_dim: Neural ODE hidden dimension.
        num_epochs: Training epochs.

    Returns:
        Dict with RMSE, relative RMSE, and other metrics.
    """
    import torch

    from reactor_twin.core.neural_ode import NeuralODE
    from reactor_twin.core.ode_func import MLPODEFunc
    from reactor_twin.utils.metrics import relative_rmse, rmse

    data = generate_synthetic_data(n_trajectories=n_train + n_test)

    # Split
    train_traj = data["trajectories"][:n_train]
    test_traj = data["trajectories"][n_train : n_train + n_test]
    train_y0 = data["y0"][:n_train]
    test_y0 = data["y0"][n_train : n_train + n_test]
    t = data["t"]

    # Build model
    ode_func = MLPODEFunc(state_dim=6, hidden_dim=hidden_dim, num_layers=3)
    model = NeuralODE(ode_func, state_dim=6)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    t_tensor = torch.tensor(t, dtype=torch.float32)

    # Training
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

    # Evaluation
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

    logger.info(f"Williams-Otto benchmark: RMSE={results['rmse']:.4f}, "
                f"Relative RMSE={results['relative_rmse']:.2f}%")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_benchmark()
    print(f"\nResults: {results}")
