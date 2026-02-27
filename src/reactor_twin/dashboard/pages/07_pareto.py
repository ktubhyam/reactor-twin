"""Pareto Front page.

Multi-objective optimization visualization: sweep one or two parameters
and plot the trade-off between two output objectives (e.g., conversion
vs temperature).
"""

from __future__ import annotations

import numpy as np
import streamlit as st

st.set_page_config(page_title="Pareto", layout="wide")
st.title("Pareto Front Analysis")


def _safe_import_plotly():
    try:
        import plotly.graph_objects as go
        return go
    except ImportError:
        return None


# ── sidebar ──────────────────────────────────────────────────────────

st.sidebar.markdown("### Reactor")
reactor_choice = st.sidebar.selectbox("Reactor", ["Exothermic A->B CSTR"])

st.sidebar.markdown("### Sweep parameter")
sweep_param = st.sidebar.selectbox("Parameter", ["F", "T_coolant", "UA"])
sp_min = st.sidebar.number_input("Min", 0.0, 1e6, 30.0)
sp_max = st.sidebar.number_input("Max", 0.0, 1e6, 300.0)
sp_n = st.sidebar.slider("Sweep points", 20, 200, 50)

st.sidebar.markdown("### Objectives")
obj1_var = st.sidebar.number_input("Objective 1 variable index", 0, 5, 0)
obj1_name = st.sidebar.text_input("Objective 1 label", "Conversion (C_A)")
obj2_var = st.sidebar.number_input("Objective 2 variable index", 0, 5, 2)
obj2_name = st.sidebar.text_input("Objective 2 label", "Temperature (T)")

t_settle = st.sidebar.slider("Settling time", 5.0, 100.0, 30.0)

# ── run ──────────────────────────────────────────────────────────────

if st.sidebar.button("Compute Pareto Front", type="primary"):
    from reactor_twin.training.data_generator import ReactorDataGenerator
    from reactor_twin.reactors.systems import create_exothermic_cstr

    vals = np.linspace(sp_min, sp_max, sp_n)
    obj1_results = []
    obj2_results = []

    progress = st.progress(0)
    for idx, v in enumerate(vals):
        reactor = create_exothermic_cstr(isothermal=False)
        reactor.params[sweep_param] = float(v)
        gen = ReactorDataGenerator(reactor)
        t_eval = np.linspace(0, t_settle, 300)
        res = gen.generate_trajectory((0, t_settle), t_eval)

        if res["success"]:
            ss = res["y"][-1]
            o1 = ss[obj1_var] if obj1_var < len(ss) else np.nan
            o2 = ss[obj2_var] if obj2_var < len(ss) else np.nan
        else:
            o1, o2 = np.nan, np.nan

        obj1_results.append(o1)
        obj2_results.append(o2)
        progress.progress(int(100 * (idx + 1) / sp_n))

    obj1_arr = np.array(obj1_results)
    obj2_arr = np.array(obj2_results)

    # Filter NaN
    valid = ~(np.isnan(obj1_arr) | np.isnan(obj2_arr))
    obj1_arr = obj1_arr[valid]
    obj2_arr = obj2_arr[valid]
    vals_valid = vals[valid]

    # Identify Pareto-optimal points (non-dominated)
    # Minimize obj1, minimize obj2
    pareto_mask = np.ones(len(obj1_arr), dtype=bool)
    for i in range(len(obj1_arr)):
        for j in range(len(obj1_arr)):
            if i == j:
                continue
            if obj1_arr[j] <= obj1_arr[i] and obj2_arr[j] <= obj2_arr[i]:
                if obj1_arr[j] < obj1_arr[i] or obj2_arr[j] < obj2_arr[i]:
                    pareto_mask[i] = False
                    break

    go = _safe_import_plotly()

    st.subheader("Trade-off Curve")
    if go is not None:
        fig = go.Figure()
        # All points
        fig.add_trace(go.Scatter(
            x=obj1_arr, y=obj2_arr, mode="markers",
            name="Operating points",
            marker=dict(size=6, color=vals_valid, colorscale="Viridis",
                        colorbar=dict(title=sweep_param)),
            text=[f"{sweep_param}={v:.1f}" for v in vals_valid],
        ))
        # Pareto front
        if pareto_mask.any():
            pareto_idx = np.where(pareto_mask)[0]
            sort_idx = np.argsort(obj1_arr[pareto_idx])
            pareto_idx = pareto_idx[sort_idx]
            fig.add_trace(go.Scatter(
                x=obj1_arr[pareto_idx], y=obj2_arr[pareto_idx],
                mode="lines+markers", name="Pareto front",
                line=dict(color="red", width=2),
                marker=dict(size=10, symbol="star"),
            ))
        fig.update_layout(xaxis_title=obj1_name, yaxis_title=obj2_name)
        st.plotly_chart(fig, use_container_width=True)
    else:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sc = ax.scatter(obj1_arr, obj2_arr, c=vals_valid, cmap="viridis", s=20)
        plt.colorbar(sc, ax=ax, label=sweep_param)
        if pareto_mask.any():
            pareto_idx = np.where(pareto_mask)[0]
            sort_idx = np.argsort(obj1_arr[pareto_idx])
            pareto_idx = pareto_idx[sort_idx]
            ax.plot(obj1_arr[pareto_idx], obj2_arr[pareto_idx], "r*-", markersize=10)
        ax.set_xlabel(obj1_name)
        ax.set_ylabel(obj2_name)
        st.pyplot(fig)

    # Pareto table
    if pareto_mask.any():
        st.subheader("Pareto-Optimal Points")
        import pandas as pd
        pidx = np.where(pareto_mask)[0]
        df = pd.DataFrame({
            sweep_param: vals_valid[pidx],
            obj1_name: obj1_arr[pidx],
            obj2_name: obj2_arr[pidx],
        })
        st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Total points", len(obj1_arr))
    col2.metric("Pareto-optimal", int(pareto_mask.sum()))

else:
    st.info("Configure settings and click **Compute Pareto Front**.")
