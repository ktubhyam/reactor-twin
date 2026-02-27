"""Sensitivity Analysis page.

One-At-a-Time (OAT) sensitivity analysis with tornado plots showing
how each parameter affects a chosen output.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

st.set_page_config(page_title="Sensitivity", layout="wide")
st.title("Sensitivity Analysis")


def _safe_import_plotly():
    try:
        import plotly.graph_objects as go

        return go
    except ImportError:
        return None


# ── sidebar ──────────────────────────────────────────────────────────

st.sidebar.markdown("### Reactor")
reactor_choice = st.sidebar.selectbox("Reactor", ["Exothermic A->B CSTR"])

st.sidebar.markdown("### OAT Settings")
perturbation = st.sidebar.slider("Perturbation (%)", 1, 50, 10) / 100.0
output_var = st.sidebar.number_input("Output variable index", 0, 5, 0)
t_settle = st.sidebar.slider("Settling time", 5.0, 100.0, 30.0)

params_to_test = st.sidebar.multiselect(
    "Parameters to test",
    ["F", "T_coolant", "UA", "V"],
    default=["F", "T_coolant", "UA"],
)

# ── run ──────────────────────────────────────────────────────────────

if st.sidebar.button("Run Sensitivity Analysis", type="primary"):
    from reactor_twin.reactors.systems import create_exothermic_cstr
    from reactor_twin.training.data_generator import ReactorDataGenerator

    labels = create_exothermic_cstr().get_state_labels()
    go = _safe_import_plotly()

    # Baseline
    reactor_base = create_exothermic_cstr(isothermal=False)
    gen_base = ReactorDataGenerator(reactor_base)
    t_eval = np.linspace(0, t_settle, 300)
    res_base = gen_base.generate_trajectory((0, t_settle), t_eval)
    base_val = res_base["y"][-1, output_var] if res_base["success"] else 0.0

    sensitivities = {}
    low_vals = {}
    high_vals = {}

    progress = st.progress(0)
    for idx, pname in enumerate(params_to_test):
        base_param = reactor_base.params.get(pname, 100.0)

        # Low perturbation
        reactor_lo = create_exothermic_cstr(isothermal=False)
        reactor_lo.params[pname] = base_param * (1 - perturbation)
        gen_lo = ReactorDataGenerator(reactor_lo)
        res_lo = gen_lo.generate_trajectory((0, t_settle), t_eval)
        lo_val = res_lo["y"][-1, output_var] if res_lo["success"] else base_val

        # High perturbation
        reactor_hi = create_exothermic_cstr(isothermal=False)
        reactor_hi.params[pname] = base_param * (1 + perturbation)
        gen_hi = ReactorDataGenerator(reactor_hi)
        res_hi = gen_hi.generate_trajectory((0, t_settle), t_eval)
        hi_val = res_hi["y"][-1, output_var] if res_hi["success"] else base_val

        sensitivities[pname] = hi_val - lo_val
        low_vals[pname] = lo_val
        high_vals[pname] = hi_val

        progress.progress(int(100 * (idx + 1) / len(params_to_test)))

    # Sort by absolute sensitivity
    sorted_params = sorted(sensitivities.keys(), key=lambda k: abs(sensitivities[k]))

    # ── Tornado plot ──
    st.subheader(f"Tornado Plot: {labels[output_var]}")

    lows = [low_vals[p] - base_val for p in sorted_params]
    highs = [high_vals[p] - base_val for p in sorted_params]

    if go is not None:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=sorted_params,
                x=lows,
                orientation="h",
                name=f"-{perturbation * 100:.0f}%",
                marker_color="steelblue",
            )
        )
        fig.add_trace(
            go.Bar(
                y=sorted_params,
                x=highs,
                orientation="h",
                name=f"+{perturbation * 100:.0f}%",
                marker_color="salmon",
            )
        )
        fig.update_layout(
            barmode="overlay",
            xaxis_title=f"Change in SS {labels[output_var]}",
            title="OAT Sensitivity (Tornado)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        y_pos = np.arange(len(sorted_params))
        ax.barh(y_pos, lows, color="steelblue", label=f"-{perturbation * 100:.0f}%")
        ax.barh(y_pos, highs, color="salmon", label=f"+{perturbation * 100:.0f}%")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_params)
        ax.set_xlabel(f"Change in SS {labels[output_var]}")
        ax.legend()
        st.pyplot(fig)

    # Sensitivity table
    st.subheader("Sensitivity Summary")
    import pandas as pd

    df = pd.DataFrame(
        {
            "Parameter": list(sensitivities.keys()),
            "Base Value": [reactor_base.params.get(p, 0) for p in sensitivities],
            f"SS {labels[output_var]} (low)": [low_vals[p] for p in sensitivities],
            f"SS {labels[output_var]} (base)": [base_val] * len(sensitivities),
            f"SS {labels[output_var]} (high)": [high_vals[p] for p in sensitivities],
            "Sensitivity": [sensitivities[p] for p in sensitivities],
        }
    )
    st.dataframe(df, use_container_width=True)

else:
    st.info("Configure settings and click **Run Sensitivity Analysis**.")
