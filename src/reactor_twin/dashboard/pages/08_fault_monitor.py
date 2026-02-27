"""Fault Monitor page.

Demonstrates the multi-level fault detection system with injected faults,
SPC control charts, residual plots, and alarm displays.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

st.set_page_config(page_title="Fault Monitor", layout="wide")
st.title("Fault Detection Monitor")


def _safe_import_plotly():
    try:
        import plotly.graph_objects as go

        return go
    except ImportError:
        return None


# ── sidebar ──────────────────────────────────────────────────────────

st.sidebar.markdown("### Simulation")
num_steps = st.sidebar.slider("Normal steps", 50, 500, 200)
fault_start = st.sidebar.slider("Fault injection at step", 10, 490, 100)
fault_magnitude = st.sidebar.slider("Fault magnitude", 0.0, 2.0, 0.5, step=0.05)
fault_variable = st.sidebar.selectbox("Fault variable index", [0, 1, 2])

st.sidebar.markdown("### SPC Settings")
ewma_lambda = st.sidebar.slider("EWMA lambda", 0.05, 0.5, 0.2, step=0.05)
cusum_h = st.sidebar.slider("CUSUM threshold h", 1.0, 10.0, 5.0, step=0.5)

# ── run ──────────────────────────────────────────────────────────────

if st.sidebar.button("Run Fault Detection", type="primary"):
    from reactor_twin.core.neural_ode import NeuralODE
    from reactor_twin.digital_twin.fault_detector import SPCChart

    state_dim = 3
    model = NeuralODE(state_dim=state_dim)

    # Generate synthetic normal data
    rng = np.random.default_rng(42)
    normal_data = rng.normal(loc=0.5, scale=0.05, size=(num_steps, state_dim))

    # Build fault data: inject a bias after fault_start
    faulty_data = normal_data.copy()
    faulty_data[fault_start:, fault_variable] += fault_magnitude

    # ── L1: SPC ──
    spc = SPCChart(num_vars=state_dim, ewma_lambda=ewma_lambda, cusum_h=cusum_h)
    spc.set_baseline(normal_data[:fault_start])

    ewma_vals = []
    cusum_pos_vals = []
    ewma_alarms = []
    cusum_alarms = []

    for i in range(num_steps):
        res = spc.update(faulty_data[i])
        ewma_vals.append(res["ewma_values"].copy())
        cusum_pos_vals.append(res["cusum_pos"].copy())
        ewma_alarms.append(res["ewma_alarm"].copy())
        cusum_alarms.append(res["cusum_alarm"].copy())

    ewma_vals = np.array(ewma_vals)
    cusum_pos_vals = np.array(cusum_pos_vals)
    ewma_alarms = np.array(ewma_alarms)
    cusum_alarms = np.array(cusum_alarms)

    go = _safe_import_plotly()
    time_axis = np.arange(num_steps)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("EWMA Chart")
        if go is not None:
            fig = go.Figure()
            for v in range(state_dim):
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=ewma_vals[:, v],
                        mode="lines",
                        name=f"Var {v}",
                    )
                )
            fig.add_vline(
                x=fault_start, line_dash="dash", line_color="red", annotation_text="Fault injected"
            )
            fig.update_layout(xaxis_title="Step", yaxis_title="EWMA value")
            st.plotly_chart(fig, use_container_width=True)
        else:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            for v in range(state_dim):
                ax.plot(time_axis, ewma_vals[:, v], label=f"Var {v}")
            ax.axvline(fault_start, color="red", linestyle="--", label="Fault")
            ax.legend()
            st.pyplot(fig)

    with col2:
        st.subheader("CUSUM Chart")
        if go is not None:
            fig2 = go.Figure()
            for v in range(state_dim):
                fig2.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=cusum_pos_vals[:, v],
                        mode="lines",
                        name=f"Var {v}",
                    )
                )
            fig2.add_hline(
                y=cusum_h, line_dash="dot", line_color="orange", annotation_text="Threshold"
            )
            fig2.add_vline(x=fault_start, line_dash="dash", line_color="red")
            fig2.update_layout(xaxis_title="Step", yaxis_title="CUSUM S+")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            import matplotlib.pyplot as plt

            fig2, ax2 = plt.subplots()
            for v in range(state_dim):
                ax2.plot(time_axis, cusum_pos_vals[:, v], label=f"Var {v}")
            ax2.axhline(cusum_h, color="orange", linestyle=":", label="Threshold")
            ax2.axvline(fault_start, color="red", linestyle="--")
            ax2.legend()
            st.pyplot(fig2)

    # ── Alarm summary ──
    st.subheader("Alarm Summary")
    first_ewma = None
    first_cusum = None
    for i in range(num_steps):
        if first_ewma is None and np.any(ewma_alarms[i]):
            first_ewma = i
        if first_cusum is None and np.any(cusum_alarms[i]):
            first_cusum = i

    c1, c2, c3 = st.columns(3)
    c1.metric("Fault injected at", f"Step {fault_start}")
    c2.metric("EWMA first alarm", f"Step {first_ewma}" if first_ewma else "None")
    c3.metric("CUSUM first alarm", f"Step {first_cusum}" if first_cusum else "None")

    detection_delay = None
    if first_ewma is not None:
        detection_delay = first_ewma - fault_start
    elif first_cusum is not None:
        detection_delay = first_cusum - fault_start

    if detection_delay is not None and detection_delay >= 0:
        st.success(f"Fault detected {detection_delay} steps after injection.")
    elif detection_delay is not None and detection_delay < 0:
        st.warning("Alarm triggered before fault injection (possible false alarm).")
    else:
        st.warning("No alarm triggered. Try increasing fault magnitude.")

else:
    st.info("Configure settings and click **Run Fault Detection**.")
