"""Phase Portrait page.

2-D vector field visualization with overlaid trajectories for
two-variable reactor systems.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

st.set_page_config(page_title="Phase Portrait", layout="wide")
st.title("Phase Portrait")


def _safe_import_plotly():
    try:
        import plotly.figure_factory as ff
        import plotly.graph_objects as go

        return ff, go
    except ImportError:
        return None, None


# ── sidebar ──────────────────────────────────────────────────────────

st.sidebar.markdown("### Reactor")
reactor_choice = st.sidebar.selectbox("Reactor", ["Exothermic A->B CSTR", "Van de Vusse CSTR"])
isothermal = st.sidebar.checkbox("Isothermal", value=True)

st.sidebar.markdown("### Phase plane axes")
x_var = st.sidebar.number_input("X variable index", 0, 5, 0)
y_var = st.sidebar.number_input("Y variable index", 0, 5, 1)

st.sidebar.markdown("### Grid resolution")
grid_n = st.sidebar.slider("Grid points per axis", 10, 30, 15)

st.sidebar.markdown("### Trajectories")
num_traj = st.sidebar.slider("Number of trajectories", 1, 10, 3)
t_end = st.sidebar.slider("Sim time", 0.5, 20.0, 5.0, step=0.5)

# ── run ──────────────────────────────────────────────────────────────

if st.sidebar.button("Generate Phase Portrait", type="primary"):
    from reactor_twin.reactors.systems import create_exothermic_cstr, create_van_de_vusse_cstr
    from reactor_twin.training.data_generator import ReactorDataGenerator

    factory = create_exothermic_cstr if "Exothermic" in reactor_choice else create_van_de_vusse_cstr
    reactor = factory(isothermal=isothermal)
    gen = ReactorDataGenerator(reactor)

    labels = reactor.get_state_labels()
    y0_default = reactor.get_initial_state()
    state_dim = reactor.state_dim

    if x_var >= state_dim or y_var >= state_dim:
        st.error(f"Variable indices must be < {state_dim}. Available: {labels}")
    else:
        # Build vector field grid
        x_center = y0_default[x_var]
        y_center = y0_default[y_var]
        x_range = max(abs(x_center) * 0.5, 0.1)
        y_range = max(abs(y_center) * 0.5, 0.1)

        xs = np.linspace(x_center - x_range, x_center + x_range, grid_n)
        ys = np.linspace(y_center - y_range, y_center + y_range, grid_n)
        X, Y = np.meshgrid(xs, ys)

        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        for i in range(grid_n):
            for j in range(grid_n):
                y_state = y0_default.copy()
                y_state[x_var] = X[i, j]
                y_state[y_var] = Y[i, j]
                dydt = reactor.ode_rhs(0.0, y_state)
                U[i, j] = dydt[x_var]
                V[i, j] = dydt[y_var]

        ff_mod, go = _safe_import_plotly()

        if ff_mod is not None and go is not None:
            # Quiver plot
            fig = ff_mod.create_quiver(
                X.tolist(),
                Y.tolist(),
                U.tolist(),
                V.tolist(),
                scale=0.02,
                arrow_scale=0.3,
                name="Vector field",
            )

            # Overlay trajectories
            t_eval = np.linspace(0, t_end, 200)
            rng = np.random.default_rng(0)
            for k in range(num_traj):
                ic = y0_default.copy()
                ic[x_var] += rng.uniform(-x_range * 0.8, x_range * 0.8)
                ic[y_var] += rng.uniform(-y_range * 0.8, y_range * 0.8)
                ic = np.maximum(ic, 0)
                result = gen.generate_trajectory((0, t_end), t_eval, y0=ic)
                if result["success"]:
                    traj = result["y"]
                    fig.add_trace(
                        go.Scatter(
                            x=traj[:, x_var],
                            y=traj[:, y_var],
                            mode="lines",
                            name=f"Traj {k + 1}",
                            line=dict(width=2),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[traj[0, x_var]],
                            y=[traj[0, y_var]],
                            mode="markers",
                            showlegend=False,
                            marker=dict(size=8, symbol="circle"),
                        )
                    )

            fig.update_layout(
                xaxis_title=labels[x_var],
                yaxis_title=labels[y_var],
                title="Phase Portrait",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.quiver(X, Y, U, V, alpha=0.5)

            t_eval = np.linspace(0, t_end, 200)
            rng = np.random.default_rng(0)
            for k in range(num_traj):
                ic = y0_default.copy()
                ic[x_var] += rng.uniform(-x_range * 0.8, x_range * 0.8)
                ic[y_var] += rng.uniform(-y_range * 0.8, y_range * 0.8)
                ic = np.maximum(ic, 0)
                result = gen.generate_trajectory((0, t_end), t_eval, y0=ic)
                if result["success"]:
                    traj = result["y"]
                    ax.plot(traj[:, x_var], traj[:, y_var], "-", lw=2, label=f"Traj {k + 1}")
                    ax.plot(traj[0, x_var], traj[0, y_var], "o")

            ax.set_xlabel(labels[x_var])
            ax.set_ylabel(labels[y_var])
            ax.legend()
            st.pyplot(fig)

else:
    st.info("Configure settings and click **Generate Phase Portrait**.")
