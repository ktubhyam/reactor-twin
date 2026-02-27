"""Parameter Sweep page.

1-D and 2-D parameter sweeps with heatmap visualization to explore
how reactor behavior depends on operating conditions.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

st.set_page_config(page_title="Parameter Sweep", layout="wide")
st.title("Parameter Sweep")


def _safe_import_plotly():
    try:
        import plotly.graph_objects as go

        return go
    except ImportError:
        return None


# ── sidebar ──────────────────────────────────────────────────────────

st.sidebar.markdown("### Reactor")
reactor_choice = st.sidebar.selectbox("Reactor", ["Exothermic A->B CSTR"])

st.sidebar.markdown("### Sweep type")
sweep_type = st.sidebar.radio("Sweep type", ["1-D", "2-D"])

st.sidebar.markdown("### Parameter 1")
param1 = st.sidebar.selectbox("Parameter 1", ["F", "T_coolant", "UA"])
p1_min = st.sidebar.number_input("P1 min", 0.0, 1e6, 50.0)
p1_max = st.sidebar.number_input("P1 max", 0.0, 1e6, 200.0)
p1_n = st.sidebar.slider("P1 points", 5, 50, 15)

if sweep_type == "2-D":
    st.sidebar.markdown("### Parameter 2")
    param2 = st.sidebar.selectbox("Parameter 2", ["T_coolant", "F", "UA"])
    p2_min = st.sidebar.number_input("P2 min", 0.0, 1e6, 280.0)
    p2_max = st.sidebar.number_input("P2 max", 0.0, 1e6, 350.0)
    p2_n = st.sidebar.slider("P2 points", 5, 50, 15)

st.sidebar.markdown("### Output")
output_var = st.sidebar.number_input("Output variable index", 0, 5, 0)
t_settle = st.sidebar.slider("Settling time", 5.0, 100.0, 30.0)

# ── run ──────────────────────────────────────────────────────────────

if st.sidebar.button("Run Sweep", type="primary"):
    from reactor_twin.reactors.systems import create_exothermic_cstr
    from reactor_twin.training.data_generator import ReactorDataGenerator

    labels = create_exothermic_cstr().get_state_labels()
    go = _safe_import_plotly()

    if sweep_type == "1-D":
        vals = np.linspace(p1_min, p1_max, p1_n)
        results = []
        progress = st.progress(0)

        for idx, v in enumerate(vals):
            reactor = create_exothermic_cstr(isothermal=False)
            reactor.params[param1] = float(v)
            gen = ReactorDataGenerator(reactor)
            t_eval = np.linspace(0, t_settle, 300)
            res = gen.generate_trajectory((0, t_settle), t_eval)
            ss = res["y"][-1, output_var] if res["success"] else np.nan
            results.append(ss)
            progress.progress(int(100 * (idx + 1) / p1_n))

        results = np.array(results)

        st.subheader(f"1-D Sweep: {param1} vs {labels[output_var]}")
        if go is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=vals, y=results, mode="lines+markers"))
            fig.update_layout(xaxis_title=param1, yaxis_title=f"SS {labels[output_var]}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot(vals, results, "o-")
            ax.set_xlabel(param1)
            ax.set_ylabel(f"SS {labels[output_var]}")
            st.pyplot(fig)

    else:  # 2-D
        v1 = np.linspace(p1_min, p1_max, p1_n)
        v2 = np.linspace(p2_min, p2_max, p2_n)
        Z = np.zeros((p2_n, p1_n))
        progress = st.progress(0)
        total = p1_n * p2_n

        for i, vv2 in enumerate(v2):
            for j, vv1 in enumerate(v1):
                reactor = create_exothermic_cstr(isothermal=False)
                reactor.params[param1] = float(vv1)
                reactor.params[param2] = float(vv2)
                gen = ReactorDataGenerator(reactor)
                t_eval = np.linspace(0, t_settle, 300)
                res = gen.generate_trajectory((0, t_settle), t_eval)
                Z[i, j] = res["y"][-1, output_var] if res["success"] else np.nan
                progress.progress(int(100 * (i * p1_n + j + 1) / total))

        st.subheader(f"2-D Heatmap: {param1} x {param2}")
        if go is not None:
            fig = go.Figure(
                data=go.Heatmap(
                    z=Z,
                    x=v1,
                    y=v2,
                    colorbar=dict(title=f"SS {labels[output_var]}"),
                )
            )
            fig.update_layout(xaxis_title=param1, yaxis_title=param2)
            st.plotly_chart(fig, use_container_width=True)
        else:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            im = ax.imshow(
                Z, extent=[p1_min, p1_max, p2_min, p2_max], aspect="auto", origin="lower"
            )
            ax.set_xlabel(param1)
            ax.set_ylabel(param2)
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)

else:
    st.info("Configure settings and click **Run Sweep**.")
