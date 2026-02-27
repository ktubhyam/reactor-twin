"""Latent Neural ODE with encoder-decoder architecture.

Demonstrates:
1. Using LatentNeuralODE to encode CSTR trajectories into a latent space
2. Evolving latent dynamics via Neural ODE
3. Decoding back to observation space
4. Showing reconstruction quality

Run: python examples/08_latent_neural_ode.py
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp

from reactor_twin import ArrheniusKinetics, CSTRReactor
from reactor_twin.core import LatentNeuralODE

np.random.seed(42)
torch.manual_seed(42)


def main() -> None:
    """Run latent Neural ODE example."""
    print("=" * 60)
    print("Example 08: Latent Neural ODE (Encoder-Decoder)")
    print("=" * 60)

    # 1. Generate CSTR data
    print("\n1. Generating CSTR training data...")
    kinetics = ArrheniusKinetics(
        name="A_to_B",
        num_reactions=1,
        params={
            "k0": np.array([1e10]),
            "Ea": np.array([50000.0]),
            "stoich": np.array([[-1, 1]]),
        },
    )
    reactor = CSTRReactor(
        name="cstr",
        num_species=2,
        params={
            "V": 100.0,
            "F": 10.0,
            "C_feed": [1.0, 0.0],
            "T_feed": 350.0,
        },
        kinetics=kinetics,
        isothermal=True,
    )

    y0 = reactor.get_initial_state()
    t_eval = np.linspace(0, 10, 40)

    sol = solve_ivp(reactor.ode_rhs, [0, 10], y0, t_eval=t_eval, method="LSODA")

    obs_dim = 2  # C_A, C_B
    z0_np = sol.y[:, 0]
    traj_np = sol.y.T  # (40, 2)

    z0 = torch.tensor(z0_np, dtype=torch.float32).unsqueeze(0)  # (1, 2)
    t_span = torch.tensor(t_eval, dtype=torch.float32)
    targets = torch.tensor(traj_np, dtype=torch.float32).unsqueeze(0)  # (1, 40, 2)

    print(f"   Observation dim: {obs_dim}")
    print(f"   Trajectory shape: {targets.shape}")
    print(f"   C_A: {traj_np[0, 0]:.3f} -> {traj_np[-1, 0]:.3f}")
    print(f"   C_B: {traj_np[0, 1]:.3f} -> {traj_np[-1, 1]:.3f}")

    # 2. Create Latent Neural ODE
    print("\n2. Creating Latent Neural ODE...")
    latent_dim = 4  # Compress to 4D latent space (from 2D obs -- over-parameterized for demo)

    model = LatentNeuralODE(
        state_dim=obs_dim,
        latent_dim=latent_dim,
        encoder_hidden_dim=32,
        decoder_hidden_dim=32,
        encoder_type="mlp",  # MLP encoder for single-point encoding
        hidden_dim=32,
        num_layers=2,
        solver="rk4",
        adjoint=False,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Obs dim: {obs_dim}, Latent dim: {latent_dim}")
    print(f"   Total parameters: {num_params:,}")
    print("   Encoder type: MLP")

    # 3. Examine latent encoding
    print("\n3. Examining latent encoding...")
    model.eval()
    with torch.no_grad():
        z_mean, z_logvar = model.encode(z0)
    print(f"   z0 (obs space): {z0[0].numpy()}")
    print(f"   z_mean (latent): {z_mean[0].numpy()}")
    print(f"   z_logvar (latent): {z_logvar[0].numpy()}")

    # 4. Train
    print("\n4. Training Latent Neural ODE...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 200

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        preds = model(z0, t_span)
        loss_dict = model.compute_loss(preds, targets)
        loss_dict["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(
                f"   Epoch {epoch + 1:>4d}/{num_epochs}: "
                f"total={loss_dict['total'].item():.6f}, "
                f"recon={loss_dict['reconstruction'].item():.6f}, "
                f"kl={loss_dict['kl'].item():.4f}"
            )

    # 5. Evaluate reconstruction
    print("\n5. Evaluating reconstruction quality...")
    model.eval()
    with torch.no_grad():
        preds = model(z0, t_span)

    pred_np = preds[0].numpy()
    true_np = traj_np

    mse = np.mean((pred_np - true_np) ** 2)
    print(f"   Reconstruction MSE: {mse:.6f}")

    print(
        f"\n   {'Time':>6} | {'True C_A':>10} | {'Pred C_A':>10} | {'True C_B':>10} | {'Pred C_B':>10}"
    )
    print("   " + "-" * 55)

    for idx in [0, 10, 20, 30, 39]:
        t = t_eval[idx]
        print(
            f"   {t:>6.1f} | {true_np[idx, 0]:>10.4f} | {pred_np[idx, 0]:>10.4f} | "
            f"{true_np[idx, 1]:>10.4f} | {pred_np[idx, 1]:>10.4f}"
        )

    # 6. Examine latent trajectories
    print("\n6. Latent trajectory analysis...")
    with torch.no_grad():
        z_mean_trained, z_logvar_trained = model.encode(z0)
        z_latent = model.reparameterize(z_mean_trained, z_logvar_trained)

    print(f"   Latent initial condition: {z_latent[0].numpy()}")
    print(f"   Latent dim: {latent_dim}, Obs dim: {obs_dim}")
    print(f"   Compression ratio: {obs_dim}/{latent_dim} = {obs_dim / latent_dim:.2f}")

    print("\n" + "=" * 60)
    print("Example 08 complete!")
    print("Key insight: LatentNeuralODE learns a compressed representation")
    print("of the dynamics. The encoder maps observations to a latent space")
    print("where a Neural ODE evolves the dynamics, and the decoder maps")
    print("back to observation space.")
    print("=" * 60)


if __name__ == "__main__":
    main()
