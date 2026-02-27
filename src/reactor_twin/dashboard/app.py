"""ReactorTwin Dashboard — main entry point.

Launch with::

    streamlit run src/reactor_twin/dashboard/app.py

Or via the installed entry-point::

    reactor-twin-dashboard
"""

from __future__ import annotations

import streamlit as st


def main() -> None:
    """Configure and launch the Streamlit dashboard."""
    st.set_page_config(
        page_title="ReactorTwin Dashboard",
        page_icon="\u2697\ufe0f",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("ReactorTwin Digital Twin Dashboard")
    st.markdown(
        "Physics-constrained Neural DEs for chemical reactor digital twins.  "
        "Select a page from the sidebar to get started."
    )

    st.sidebar.success("Select a page above.")

    st.markdown("---")
    st.markdown(
        "### Available Pages\n"
        "| Page | Description |\n"
        "|------|-------------|\n"
        "| **Reactor Sim** | Configure and simulate reactor dynamics |\n"
        "| **Phase Portrait** | 2-D vector field and trajectory visualization |\n"
        "| **Bifurcation** | Parameter sweep and steady-state diagrams |\n"
        "| **RTD Analysis** | Residence time distribution curves |\n"
        "| **Parameter Sweep** | 1-D / 2-D heatmap sweeps |\n"
        "| **Sensitivity** | Tornado plots and OAT sensitivity |\n"
        "| **Pareto** | Multi-objective optimization fronts |\n"
        "| **Fault Monitor** | SPC charts, residual alarms, fault isolation |\n"
        "| **Model Validation** | Parity plots and error metrics |\n"
        "| **Latent Explorer** | 2-D / 3-D latent space visualization |"
    )


if __name__ == "__main__":
    main()
