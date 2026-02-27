"""Latent Explorer page.

Visualize the latent space of a LatentNeuralODE model in 2-D or 3-D.
Encodes reactor trajectories into the learned latent representation and
plots them colored by time or initial condition.
"""

from __future__ import annotations

import numpy as np
import streamlit as st
import torch

st.set_page_config(page_title="Latent Explorer", layout="wide")
st.title("Latent Space Explorer")


def _safe_import_plotly():
    try:
        import plotly.graph_objects as go

        return go
    except ImportError:
        return None


# ── sidebar ──────────────────────────────────────────────────────────

st.sidebar.markdown("### Latent ODE Config")
latent_dim = st.sidebar.selectbox("Latent dimension", [2, 3, 4, 8], index=0)
encoder_type = st.sidebar.selectbox("Encoder type", ["gru", "mlp"])

st.sidebar.markdown("### Training")
hidden_dim = st.sidebar.selectbox("Hidden dim", [32, 64], index=1)
num_epochs = st.sidebar.slider("Training epochs", 5, 100, 30)

st.sidebar.markdown("### Visualization")
num_trajectories = st.sidebar.slider("Trajectories to visualize", 5, 50, 15)
t_end = st.sidebar.slider("Sim time", 0.5, 5.0, 2.0, step=0.5)
num_points = st.sidebar.slider("Time points", 20, 100, 40)
plot_3d = st.sidebar.checkbox("3-D plot (if latent_dim >= 3)", value=False)

# ── run ──────────────────────────────────────────────────────────────

if st.sidebar.button("Explore Latent Space", type="primary"):
    from reactor_twin.core.latent_neural_ode import LatentNeuralODE
    from reactor_twin.reactors.systems import create_exothermic_cstr
    from reactor_twin.training.data_generator import ReactorDataGenerator

    progress = st.progress(0, text="Setting up...")

    reactor = create_exothermic_cstr(isothermal=False)
    gen = ReactorDataGenerator(reactor)
    t_eval = np.linspace(0, t_end, num_points)
    state_dim = reactor.state_dim

    progress.progress(10, text="Generating data...")
    train_data = gen.generate_dataset(60, (0, t_end), t_eval, batch_size=16)

    # Create LatentNeuralODE
    model = LatentNeuralODE(
        state_dim=state_dim,
        latent_dim=latent_dim,
        encoder_hidden_dim=hidden_dim,
        decoder_hidden_dim=hidden_dim,
        encoder_type=encoder_type,
        adjoint=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training
    progress.progress(15, text="Training Latent Neural ODE...")
    model.train()
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in train_data:
            step_losses = model.train_step(batch, optimizer)
            epoch_loss += step_losses["total"]
        epoch_loss /= len(train_data)
        losses.append(epoch_loss)
        pct = 15 + int(75 * (epoch + 1) / num_epochs)
        progress.progress(pct, text=f"Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f}")

    progress.progress(92, text="Encoding trajectories...")

    # Generate visualization trajectories
    vis_data = gen.generate_batch(num_trajectories, (0, t_end), t_eval)
    z0_vis = vis_data["z0"]  # (N, state_dim)
    t_tensor = vis_data["t_span"]

    model.eval()
    with torch.no_grad():
        # Encode initial observations to latent space
        mu, logvar = model.encode(z0_vis)
        z_latent_0 = model.reparameterize(mu, logvar)  # (N, latent_dim)

        # Integrate in latent space to get latent trajectories
        ode_func = model.ode_func
        from torchdiffeq import odeint

        z_traj = odeint(ode_func, z_latent_0, t_tensor, method="euler")
        # z_traj shape: (T, N, latent_dim) -> transpose to (N, T, latent_dim)
        z_traj = z_traj.transpose(0, 1)

    z_traj_np = z_traj.numpy()  # (N, T, latent_dim)
    progress.progress(100, text="Done!")

    go = _safe_import_plotly()

    # ── Loss curve ──
    st.subheader("Training Loss")
    if go is not None:
        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(x=list(range(1, num_epochs + 1)), y=losses, mode="lines"))
        fig_l.update_layout(xaxis_title="Epoch", yaxis_title="Loss", yaxis_type="log")
        st.plotly_chart(fig_l, use_container_width=True)
    else:
        import matplotlib.pyplot as plt

        fig_l, ax_l = plt.subplots()
        ax_l.semilogy(range(1, num_epochs + 1), losses)
        st.pyplot(fig_l)

    # ── Latent space visualization ──
    st.subheader("Latent Space Trajectories")

    if plot_3d and latent_dim >= 3 and go is not None:
        fig3d = go.Figure()
        for i in range(num_trajectories):
            fig3d.add_trace(
                go.Scatter3d(
                    x=z_traj_np[i, :, 0],
                    y=z_traj_np[i, :, 1],
                    z=z_traj_np[i, :, 2],
                    mode="lines",
                    name=f"Traj {i}",
                    line=dict(width=2),
                )
            )
        fig3d.update_layout(
            scene=dict(xaxis_title="z1", yaxis_title="z2", zaxis_title="z3"),
            title="3-D Latent Trajectories",
        )
        st.plotly_chart(fig3d, use_container_width=True)
    # 2-D plot (first two latent dims)
    elif go is not None:
        fig2d = go.Figure()
        for i in range(num_trajectories):
            fig2d.add_trace(
                go.Scatter(
                    x=z_traj_np[i, :, 0],
                    y=z_traj_np[i, :, 1],
                    mode="lines+markers",
                    name=f"Traj {i}",
                    marker=dict(size=3),
                )
            )
        fig2d.update_layout(
            xaxis_title="z1",
            yaxis_title="z2",
            title="2-D Latent Trajectories",
        )
        st.plotly_chart(fig2d, use_container_width=True)
    else:
        import matplotlib.pyplot as plt

        fig2d, ax = plt.subplots()
        for i in range(num_trajectories):
            ax.plot(z_traj_np[i, :, 0], z_traj_np[i, :, 1], "-o", markersize=2)
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_title("2-D Latent Trajectories")
        st.pyplot(fig2d)

    # ── Latent initial conditions scatter ──
    st.subheader("Initial Latent Encodings")
    mu_np = mu.numpy()
    if go is not None:
        fig_ic = go.Figure()
        fig_ic.add_trace(
            go.Scatter(
                x=mu_np[:, 0],
                y=mu_np[:, 1] if latent_dim > 1 else np.zeros(num_trajectories),
                mode="markers",
                marker=dict(size=8, color=np.arange(num_trajectories), colorscale="Viridis"),
            )
        )
        fig_ic.update_layout(
            xaxis_title="mu_1", yaxis_title="mu_2", title="Encoded Initial States (mean)"
        )
        st.plotly_chart(fig_ic, use_container_width=True)
    else:
        import matplotlib.pyplot as plt

        fig_ic, ax_ic = plt.subplots()
        ax_ic.scatter(
            mu_np[:, 0],
            mu_np[:, 1] if latent_dim > 1 else np.zeros(num_trajectories),
            c=np.arange(num_trajectories),
            cmap="viridis",
        )
        ax_ic.set_xlabel("mu_1")
        ax_ic.set_ylabel("mu_2")
        st.pyplot(fig_ic)

else:
    st.info("Configure settings and click **Explore Latent Space**.")
