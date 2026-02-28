"""ReactorTwin FastAPI server for real-time reactor simulation.

Requires the [api] optional dependencies:
    pip install reactor-twin[api]
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException, Query, WebSocket
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError as exc:
    raise ImportError(
        "FastAPI is required for the API server. Install with: pip install reactor-twin[api]"
    ) from exc

import numpy as np

from reactor_twin.reactors.systems import (
    create_exothermic_cstr,
    create_van_de_vusse_cstr,
)

# ── Pydantic response models ─────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str


class ReactorListResponse(BaseModel):
    reactors: list[str]


class SimulationResponse(BaseModel):
    time: list[float]
    states: list[list[float]]
    labels: list[str]
    success: bool


class ErrorResponse(BaseModel):
    detail: str


# ── Application ──────────────────────────────────────────────────────

app = FastAPI(
    title="ReactorTwin API",
    description="Physics-constrained Neural DE reactor simulation API",
    version="0.3.0",
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


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.get("/reactors", response_model=ReactorListResponse)
def list_reactors() -> ReactorListResponse:
    """List available benchmark reactors."""
    return ReactorListResponse(reactors=list(BENCHMARKS.keys()))


@app.post(
    "/simulate/{reactor_name}",
    response_model=SimulationResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def simulate(
    reactor_name: str,
    t_end: float = Query(default=10.0, gt=0, le=1000),
    num_points: int = Query(default=100, ge=2, le=10000),
) -> SimulationResponse:
    """Simulate a benchmark reactor.

    Args:
        reactor_name: Name of the benchmark reactor.
        t_end: Simulation end time (0, 1000].
        num_points: Number of output time points [2, 10000].

    Returns:
        SimulationResponse with time, states, labels, and success flag.
    """
    if reactor_name not in BENCHMARKS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown reactor: {reactor_name}. Available: {list(BENCHMARKS.keys())}",
        )

    from scipy.integrate import solve_ivp

    reactor = BENCHMARKS[reactor_name]()
    y0 = reactor.get_initial_state()
    t_eval = np.linspace(0, t_end, num_points)

    try:
        sol = solve_ivp(
            reactor.ode_rhs,
            [0, t_end],
            y0,
            t_eval=t_eval,
            method="RK45",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Solver failed: {exc}") from exc

    if not sol.success:
        raise HTTPException(status_code=500, detail=f"Solver failed: {sol.message}")

    return SimulationResponse(
        time=sol.t.tolist(),
        states=sol.y.T.tolist(),
        labels=reactor.get_state_labels(),
        success=sol.success,
    )


# ── WebSocket ────────────────────────────────────────────────────────


@app.websocket("/ws/simulate")
async def websocket_simulate(websocket: WebSocket) -> None:
    """Stream simulation results over WebSocket."""
    from reactor_twin.api.websocket import simulate_ws

    await simulate_ws(websocket, BENCHMARKS)


# ── Entry point ──────────────────────────────────────────────────────


def main() -> None:
    """Entry point for reactor-twin-api command."""
    import uvicorn

    logger.info("Starting ReactorTwin API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)


__all__ = ["app", "main"]
