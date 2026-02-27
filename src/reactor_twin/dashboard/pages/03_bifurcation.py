"""Bifurcation Diagram page.

Sweep a single reactor parameter and find steady-state values to
construct a bifurcation diagram.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

st.set_page_config(page_title="Bifurcation", layout="wide")
st.title("Bifurcation Diagram")


def _safe_import_plotly():
    try:
        import plotly.graph_objects as go
        return go
    except ImportError:
        return None


# ── sidebar ──────────────────────────────────────────────────────────

st.sidebar.markdown("### Reactor")
reactor_choice = st.sidebar.selectbox("Reactor", ["Exothermic A->B CSTR"])

st.sidebar.markdown("### Bifurcation parameter")
param_name = st.sidebar.selectbox("Parameter to sweep", ["F", "T_coolant", "UA"])
param_min = st.sidebar.number_input("Min value", 0.0, 1e6, 50.0)
param_max = st.sidebar.number_input("Max value", 0.0, 1e6, 300.0)
num_sweep = st.sidebar.slider("Sweep points", 20, 200, 60)

st.sidebar.markdown("### Steady-state detection")
t_settle = st.sidebar.slider("Settling time", 10.0, 200.0, 50.0)
output_var = st.sidebar.number_input("Output variable index", 0, 5, 0)

# ── run ──────────────────────────────────────────────────────────────

if st.sidebar.button("Compute Bifurcation", type="primary"):
    from reactor_twin.training.data_generator import ReactorDataGenerator
    from reactor_twin.reactors.systems import create_exothermic_cstr

    progress = st.progress(0, text="Sweeping parameter...")

    param_values = np.linspace(param_min, param_max, num_sweep)
    steady_states = []

    for idx, pval in enumerate(param_values):
        reactor = create_exothermic_cstr(isothermal=False)
        reactor.params[param_name] = float(pval)

        gen = ReactorDataGenerator(reactor)
        t_eval = np.linspace(0, t_settle, 500)
        result = gen.generate_trajectory((0, t_settle), t_eval)

        if result["success"]:
            ss = result["y"][-1, output_var]
        else:
            ss = np.nan
        steady_states.append(ss)

        pct = int(100 * (idx + 1) / num_sweep)
        progress.progress(pct, text=f"Point {idx+1}/{num_sweep}")

    progress.progress(100, text="Done!")
    steady_states = np.array(steady_states)
    labels = create_exothermic_cstr().get_state_labels()

    go = _safe_import_plotly()

    st.subheader("Steady-State Bifurcation Diagram")
    if go is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=param_values, y=steady_states,
            mode="markers+lines", name=labels[output_var],
            marker=dict(size=5),
        ))
        fig.update_layout(
            xaxis_title=param_name,
            yaxis_title=f"Steady-state {labels[output_var]}",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(param_values, steady_states, "o-", markersize=3)
        ax.set_xlabel(param_name)
        ax.set_ylabel(f"Steady-state {labels[output_var]}")
        st.pyplot(fig)

    # Show raw data
    with st.expander("Raw data"):
        import pandas as pd
        df = pd.DataFrame({param_name: param_values, f"SS {labels[output_var]}": steady_states})
        st.dataframe(df, use_container_width=True)

else:
    st.info("Configure settings and click **Compute Bifurcation**.")
