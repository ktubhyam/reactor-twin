"""Model Validation page.

Compare Neural ODE predictions against ground-truth reactor trajectories.
Shows parity plots, prediction overlays, and error metrics.
"""

from __future__ import annotations

import numpy as np
import streamlit as st
import torch

st.set_page_config(page_title="Model Validation", layout="wide")
st.title("Model Validation")


def _safe_import_plotly():
    try:
        import plotly.graph_objects as go

        return go
    except ImportError:
        return None


# ── sidebar ──────────────────────────────────────────────────────────

st.sidebar.markdown("### Reactor")
reactor_choice = st.sidebar.selectbox("Reactor", ["Exothermic A->B CSTR", "Van de Vusse CSTR"])
isothermal = st.sidebar.checkbox("Isothermal", value=False)

st.sidebar.markdown("### Training")
hidden_dim = st.sidebar.selectbox("Hidden dim", [32, 64, 128], index=1)
num_epochs = st.sidebar.slider("Training epochs", 5, 100, 20)
num_train_traj = st.sidebar.slider("Training trajectories", 10, 200, 50)

st.sidebar.markdown("### Simulation")
t_end = st.sidebar.slider("Sim time", 0.5, 10.0, 2.0, step=0.5)
num_points = st.sidebar.slider("Time points", 20, 200, 50)

# ── run ──────────────────────────────────────────────────────────────

if st.sidebar.button("Train & Validate", type="primary"):
    from reactor_twin.core.neural_ode import NeuralODE
    from reactor_twin.reactors.systems import create_exothermic_cstr, create_van_de_vusse_cstr
    from reactor_twin.training.data_generator import ReactorDataGenerator

    progress = st.progress(0, text="Creating reactor...")
    factory = create_exothermic_cstr if "Exothermic" in reactor_choice else create_van_de_vusse_cstr
    reactor = factory(isothermal=isothermal)

    gen = ReactorDataGenerator(reactor)
    t_eval = np.linspace(0, t_end, num_points)

    progress.progress(10, text="Generating training data...")
    train_data = gen.generate_dataset(num_train_traj, (0, t_end), t_eval, batch_size=16)

    # Create model
    model = NeuralODE(state_dim=reactor.state_dim, hidden_dim=hidden_dim, adjoint=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    progress.progress(20, text="Training model...")
    model.train()
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in train_data:
            step_losses = model.train_step(batch, optimizer)
            epoch_loss += step_losses["total"]
        epoch_loss /= len(train_data)
        losses.append(epoch_loss)
        pct = 20 + int(70 * (epoch + 1) / num_epochs)
        progress.progress(pct, text=f"Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.6f}")

    progress.progress(95, text="Generating validation trajectories...")

    # Validation trajectory
    val_result = gen.generate_trajectory((0, t_end), t_eval)
    y_true = val_result["y"]  # (num_points, state_dim)
    t = val_result["t"]

    z0 = torch.tensor(y_true[0], dtype=torch.float32).unsqueeze(0)
    t_tensor = torch.tensor(t_eval, dtype=torch.float32)
    y_pred = model.predict(z0, t_tensor).squeeze(0).numpy()

    progress.progress(100, text="Done!")
    labels = reactor.get_state_labels()

    go = _safe_import_plotly()

    # ── Training loss curve ──
    st.subheader("Training Loss")
    if go is not None:
        fig_loss = go.Figure()
        fig_loss.add_trace(
            go.Scatter(
                x=list(range(1, num_epochs + 1)),
                y=losses,
                mode="lines+markers",
                name="Train loss",
            )
        )
        fig_loss.update_layout(xaxis_title="Epoch", yaxis_title="MSE Loss", yaxis_type="log")
        st.plotly_chart(fig_loss, use_container_width=True)
    else:
        import matplotlib.pyplot as plt

        fig_l, ax_l = plt.subplots()
        ax_l.semilogy(range(1, num_epochs + 1), losses)
        ax_l.set_xlabel("Epoch")
        ax_l.set_ylabel("MSE Loss")
        st.pyplot(fig_l)

    # ── Prediction vs Ground Truth overlay ──
    st.subheader("Prediction vs Ground Truth")
    cols = st.columns(min(reactor.state_dim, 3))
    for i, col in enumerate(cols):
        if i >= reactor.state_dim:
            break
        with col:
            if go is not None:
                fig_ov = go.Figure()
                fig_ov.add_trace(go.Scatter(x=t, y=y_true[:, i], mode="lines", name="Truth"))
                fig_ov.add_trace(
                    go.Scatter(
                        x=t, y=y_pred[:, i], mode="lines", name="Predicted", line=dict(dash="dash")
                    )
                )
                fig_ov.update_layout(title=labels[i], xaxis_title="Time", yaxis_title=labels[i])
                st.plotly_chart(fig_ov, use_container_width=True)
            else:
                import matplotlib.pyplot as plt

                fig_o, ax_o = plt.subplots()
                ax_o.plot(t, y_true[:, i], label="Truth")
                ax_o.plot(t, y_pred[:, i], "--", label="Predicted")
                ax_o.set_title(labels[i])
                ax_o.legend()
                st.pyplot(fig_o)

    # ── Parity plots ──
    st.subheader("Parity Plots")
    cols2 = st.columns(min(reactor.state_dim, 3))
    for i, col in enumerate(cols2):
        if i >= reactor.state_dim:
            break
        with col:
            if go is not None:
                lo = min(y_true[:, i].min(), y_pred[:, i].min())
                hi = max(y_true[:, i].max(), y_pred[:, i].max())
                fig_p = go.Figure()
                fig_p.add_trace(
                    go.Scatter(
                        x=y_true[:, i],
                        y=y_pred[:, i],
                        mode="markers",
                        name=labels[i],
                    )
                )
                fig_p.add_trace(
                    go.Scatter(
                        x=[lo, hi],
                        y=[lo, hi],
                        mode="lines",
                        name="y=x",
                        line=dict(dash="dot", color="gray"),
                    )
                )
                fig_p.update_layout(title=labels[i], xaxis_title="True", yaxis_title="Predicted")
                st.plotly_chart(fig_p, use_container_width=True)
            else:
                import matplotlib.pyplot as plt

                fig_pp, ax_pp = plt.subplots()
                ax_pp.scatter(y_true[:, i], y_pred[:, i], s=10)
                lims = [
                    min(y_true[:, i].min(), y_pred[:, i].min()),
                    max(y_true[:, i].max(), y_pred[:, i].max()),
                ]
                ax_pp.plot(lims, lims, "k--", alpha=0.5)
                ax_pp.set_title(labels[i])
                ax_pp.set_xlabel("True")
                ax_pp.set_ylabel("Predicted")
                st.pyplot(fig_pp)

    # ── Error metrics ──
    st.subheader("Error Metrics")
    errors = y_true - y_pred
    mse_per_var = np.mean(errors**2, axis=0)
    mae_per_var = np.mean(np.abs(errors), axis=0)
    max_err_per_var = np.max(np.abs(errors), axis=0)

    import pandas as pd

    metrics_df = pd.DataFrame(
        {
            "Variable": labels,
            "MSE": mse_per_var,
            "MAE": mae_per_var,
            "Max Abs Error": max_err_per_var,
        }
    )
    st.dataframe(metrics_df, use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric("Overall MSE", f"{np.mean(mse_per_var):.6f}")
    m2.metric("Overall MAE", f"{np.mean(mae_per_var):.6f}")

else:
    st.info("Configure settings and click **Train & Validate**.")
