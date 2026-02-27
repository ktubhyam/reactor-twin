"""RTD Analysis page.

Compute and visualize the Residence Time Distribution E(t) curve
for a reactor using a tracer pulse injection.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

st.set_page_config(page_title="RTD Analysis", layout="wide")
st.title("Residence Time Distribution Analysis")


def _safe_import_plotly():
    try:
        import plotly.graph_objects as go
        return go
    except ImportError:
        return None


# ── sidebar ──────────────────────────────────────────────────────────

st.sidebar.markdown("### Reactor")
reactor_choice = st.sidebar.selectbox("Reactor", ["Exothermic A->B CSTR"])
isothermal = st.sidebar.checkbox("Isothermal", value=True)

st.sidebar.markdown("### RTD Settings")
t_end = st.sidebar.slider("Observation time", 1.0, 50.0, 10.0)
num_points = st.sidebar.slider("Time points", 100, 1000, 500)
tracer_pulse = st.sidebar.number_input("Tracer pulse concentration", 0.1, 10.0, 1.0)

# ── theory ───────────────────────────────────────────────────────────

st.markdown("""
### Background

The **Residence Time Distribution** *E(t)* describes how long fluid elements
spend inside a reactor.  For an ideal CSTR with volume *V* and flow rate *F*,
the mean residence time is *tau = V / F* and:

    E(t) = (1/tau) * exp(-t/tau)

Below we simulate the tracer washout curve from the reactor ODE and compare
it to the ideal CSTR RTD.
""")

# ── run ──────────────────────────────────────────────────────────────

if st.sidebar.button("Compute RTD", type="primary"):
    from reactor_twin.training.data_generator import ReactorDataGenerator
    from reactor_twin.reactors.systems import create_exothermic_cstr

    reactor = create_exothermic_cstr(isothermal=isothermal)
    gen = ReactorDataGenerator(reactor)

    # Inject tracer: set C_A(0) = tracer_pulse, C_A_feed = 0
    y0 = reactor.get_initial_state().copy()
    y0[0] = tracer_pulse  # species A as tracer
    original_feed = list(reactor.params["C_feed"])
    reactor.params["C_feed"] = [0.0] + original_feed[1:]

    t_eval = np.linspace(0, t_end, num_points)
    result = gen.generate_trajectory((0, t_end), t_eval, y0=y0)

    if not result["success"]:
        st.error("Integration failed.")
    else:
        t = result["t"]
        C_tracer = result["y"][:, 0]  # Species A concentration

        # Normalize to get E(t)
        area = np.trapz(C_tracer, t)
        if area > 1e-10:
            E_t = C_tracer / area
        else:
            E_t = np.zeros_like(C_tracer)

        # Ideal CSTR RTD
        V = reactor.params.get("V", 100.0)
        F = reactor.params.get("F", 100.0)
        tau = V / F
        E_ideal = (1.0 / tau) * np.exp(-t / tau)

        go = _safe_import_plotly()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Tracer Washout Curve")
            if go is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=t, y=C_tracer, mode="lines", name="C_tracer(t)"))
                fig.update_layout(xaxis_title="Time", yaxis_title="Tracer Concentration")
                st.plotly_chart(fig, use_container_width=True)
            else:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.plot(t, C_tracer)
                ax.set_xlabel("Time")
                ax.set_ylabel("Tracer Concentration")
                st.pyplot(fig)

        with col2:
            st.subheader("E(t) Curve")
            if go is not None:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=t, y=E_t, mode="lines", name="E(t) simulated"))
                fig2.add_trace(go.Scatter(x=t, y=E_ideal, mode="lines", name="E(t) ideal CSTR",
                                          line=dict(dash="dash")))
                fig2.update_layout(xaxis_title="Time", yaxis_title="E(t)")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                import matplotlib.pyplot as plt
                fig2, ax2 = plt.subplots()
                ax2.plot(t, E_t, label="Simulated")
                ax2.plot(t, E_ideal, "--", label="Ideal CSTR")
                ax2.set_xlabel("Time")
                ax2.set_ylabel("E(t)")
                ax2.legend()
                st.pyplot(fig2)

        # Metrics
        mean_rt = np.trapz(t * E_t, t)
        variance = np.trapz((t - mean_rt) ** 2 * E_t, t)

        m1, m2, m3 = st.columns(3)
        m1.metric("Mean residence time", f"{mean_rt:.3f}")
        m2.metric("Theoretical tau (V/F)", f"{tau:.3f}")
        m3.metric("Variance", f"{variance:.3f}")

else:
    st.info("Configure settings and click **Compute RTD**.")
