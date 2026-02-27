"""ReactorTwin FastAPI server for real-time reactor simulation.

Requires the [api] optional dependencies:
    pip install reactor-twin[api]
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
except ImportError as exc:
    raise ImportError(
        "FastAPI is required for the API server. "
        "Install with: pip install reactor-twin[api]"
    ) from exc

import numpy as np

from reactor_twin.reactors.systems import (
    create_exothermic_cstr,
    create_van_de_vusse_cstr,
)

app = FastAPI(
    title="ReactorTwin API",
    description="Physics-constrained Neural DE reactor simulation API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pre-built reactor benchmarks
BENCHMARKS = {
    "exothermic_ab": create_exothermic_cstr,
    "van_de_vusse": create_van_de_vusse_cstr,
}


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/reactors")
def list_reactors() -> dict[str, list[str]]:
    """List available benchmark reactors."""
    return {"reactors": list(BENCHMARKS.keys())}


@app.post("/simulate/{reactor_name}")
def simulate(
    reactor_name: str,
    t_end: float = 10.0,
    num_points: int = 100,
) -> dict:
    """Simulate a benchmark reactor.

    Args:
        reactor_name: Name of the benchmark reactor.
        t_end: Simulation end time.
        num_points: Number of output time points.

    Returns:
        Dictionary with time, states, and labels.
    """
    if reactor_name not in BENCHMARKS:
        return {"error": f"Unknown reactor: {reactor_name}. Available: {list(BENCHMARKS.keys())}"}

    from scipy.integrate import solve_ivp

    reactor = BENCHMARKS[reactor_name]()
    y0 = reactor.get_initial_state()
    t_eval = np.linspace(0, t_end, num_points)

    sol = solve_ivp(
        reactor.ode_rhs,
        [0, t_end],
        y0,
        t_eval=t_eval,
        method="RK45",
    )

    return {
        "time": sol.t.tolist(),
        "states": sol.y.T.tolist(),
        "labels": reactor.get_state_labels(),
        "success": sol.success,
    }


def main() -> None:
    """Entry point for reactor-twin-api command."""
    import uvicorn

    logger.info("Starting ReactorTwin API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)


__all__ = ["app", "main"]
