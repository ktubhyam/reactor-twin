"""Reactor Simulation page.

Select a reactor type, configure parameters, run a scipy simulation,
and plot concentration / temperature time-series.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

st.set_page_config(page_title="Reactor Sim", layout="wide")
st.title("Reactor Simulation")


# ── helpers ──────────────────────────────────────────────────────────

def _get_reactor_factories() -> dict:
    """Return available benchmark reactor factory functions."""
    from reactor_twin.reactors.systems import (
        create_exothermic_cstr,
        create_van_de_vusse_cstr,
    )
    return {
        "Exothermic A->B CSTR": create_exothermic_cstr,
        "Van de Vusse CSTR": create_van_de_vusse_cstr,
    }


def _safe_import_plotly():
    try:
        import plotly.graph_objects as go
        return go
    except ImportError:
        return None


# ── sidebar ──────────────────────────────────────────────────────────

factories = _get_reactor_factories()
reactor_name = st.sidebar.selectbox("Reactor type", list(factories.keys()))

st.sidebar.markdown("### Simulation settings")
t_end = st.sidebar.slider("Simulation time", 0.1, 20.0, 5.0, step=0.1)
num_points = st.sidebar.slider("Time points", 50, 500, 200)
isothermal = st.sidebar.checkbox("Isothermal", value=False)

# ── parameter overrides ─────────────────────────────────────────────

st.sidebar.markdown("### Parameter overrides")
if "Exothermic" in reactor_name:
    F_val = st.sidebar.number_input("Feed flow F (L/min)", 10.0, 500.0, 100.0)
    T_cool = st.sidebar.number_input("Coolant T (K)", 250.0, 400.0, 300.0)
else:
    F_val = st.sidebar.number_input("Feed flow F (L/min)", 1.0, 500.0, 100.0)
    T_cool = st.sidebar.number_input("Coolant T (K)", 250.0, 400.0, 300.0)

# ── simulation ───────────────────────────────────────────────────────

if st.sidebar.button("Run Simulation", type="primary"):
    from reactor_twin.training.data_generator import ReactorDataGenerator

    factory = factories[reactor_name]
    reactor = factory(isothermal=isothermal)

    # Apply overrides
    reactor.params["F"] = F_val
    if not isothermal and "T_coolant" in reactor.params:
        reactor.params["T_coolant"] = T_cool

    gen = ReactorDataGenerator(reactor)
    t_eval = np.linspace(0, t_end, num_points)
    result = gen.generate_trajectory((0, t_end), t_eval)

    if not result["success"]:
        st.error("Integration failed. Try reducing simulation time or changing parameters.")
    else:
        t = result["t"]
        y = result["y"]
        labels = reactor.get_state_labels()

        go = _safe_import_plotly()

        # Time-series plot
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Concentrations")
            if go is not None:
                fig = go.Figure()
                for i in range(reactor.num_species):
                    fig.add_trace(go.Scatter(x=t, y=y[:, i], mode="lines", name=labels[i]))
                fig.update_layout(xaxis_title="Time", yaxis_title="Concentration (mol/L)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                for i in range(reactor.num_species):
                    ax.plot(t, y[:, i], label=labels[i])
                ax.set_xlabel("Time")
                ax.set_ylabel("Concentration (mol/L)")
                ax.legend()
                st.pyplot(fig)

        with col2:
            if not isothermal and y.shape[1] > reactor.num_species:
                st.subheader("Temperature")
                temp_idx = reactor.num_species
                if go is not None:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=t, y=y[:, temp_idx], mode="lines", name=labels[temp_idx]))
                    fig2.update_layout(xaxis_title="Time", yaxis_title="Temperature (K)")
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    import matplotlib.pyplot as plt
                    fig2, ax2 = plt.subplots()
                    ax2.plot(t, y[:, temp_idx], label=labels[temp_idx])
                    ax2.set_xlabel("Time")
                    ax2.set_ylabel("Temperature (K)")
                    ax2.legend()
                    st.pyplot(fig2)

        # Raw data
        with st.expander("Raw data"):
            import pandas as pd
            df = pd.DataFrame(y, columns=labels)
            df.insert(0, "time", t)
            st.dataframe(df, use_container_width=True)
else:
    st.info("Configure parameters in the sidebar and click **Run Simulation**.")
