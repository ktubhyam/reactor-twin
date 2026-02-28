"""ReactorTwin FastAPI server for real-time reactor simulation.

Requires the [api] optional dependencies:
    pip install reactor-twin[api]
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError as exc:
    raise ImportError(
        "FastAPI is required for the API server. Install with: pip install reactor-twin[api]"
    ) from exc

import numpy as np

from reactor_twin.api.metrics import make_metrics_app
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
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_metrics_app = make_metrics_app()
if _metrics_app is not None:
    app.mount("/metrics", _metrics_app)

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


# ── API v2: Model Serving ────────────────────────────────────────────

import tempfile
from pathlib import Path

import torch

from reactor_twin.api.auth import create_token, rate_limiter, require_auth


class TokenRequest(BaseModel):
    subject: str


class TokenResponse(BaseModel):
    token: str


class ModelUploadResponse(BaseModel):
    model_id: str
    message: str


class PredictionRequest(BaseModel):
    z0: list[float]
    t_span: list[float]
    controls: list[list[float]] | None = None


class PredictionResponse(BaseModel):
    trajectory: list[list[float]]
    success: bool


class BatchPredictionRequest(BaseModel):
    samples: list[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]
    success: bool


# In-memory model store for v2
_loaded_models: dict[str, torch.nn.Module] = {}


@app.post("/api/v2/token", response_model=TokenResponse)
def get_token(req: TokenRequest) -> TokenResponse:
    """Generate a JWT token for API access."""
    token = create_token(req.subject)
    return TokenResponse(token=token)


@app.post(
    "/api/v2/models/upload",
    response_model=ModelUploadResponse,
    responses={401: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
)
async def upload_model(
    request: Request,
) -> ModelUploadResponse:
    """Upload a PyTorch model checkpoint.

    Expects the raw bytes of a ``torch.save()`` checkpoint in the
    request body.  The model is loaded into memory and assigned an ID.
    """
    rate_limiter.check(request)
    user = await require_auth(request)

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty request body")

    # Save to temp file and load
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        f.write(body)
        tmp_path = f.name

    try:
        checkpoint = torch.load(tmp_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid checkpoint: {exc}") from exc

    import hashlib

    model_id = hashlib.sha256(body[:1024]).hexdigest()[:12]
    _loaded_models[model_id] = checkpoint
    Path(tmp_path).unlink(missing_ok=True)

    logger.info(f"Model uploaded: id={model_id}, by={user.get('sub')}")
    return ModelUploadResponse(model_id=model_id, message="Model uploaded successfully")


@app.post(
    "/api/v2/models/{model_id}/predict",
    response_model=PredictionResponse,
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
)
async def predict(
    model_id: str,
    req: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    """Run a single prediction using an uploaded model.

    The model must have been previously uploaded via ``/api/v2/models/upload``.
    """
    rate_limiter.check(request)
    await require_auth(request)

    if model_id not in _loaded_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    model = _loaded_models[model_id]

    try:
        z0 = torch.tensor(req.z0, dtype=torch.float32).unsqueeze(0)
        t_span = torch.tensor(req.t_span, dtype=torch.float32)

        if hasattr(model, "forward"):
            controls = None
            if req.controls is not None:
                controls = torch.tensor(req.controls, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                result = model(z0=z0, t_span=t_span, controls=controls)
            trajectory = result.squeeze(0).tolist()
        else:
            # Raw state dict — can't predict, but return placeholder
            raise HTTPException(
                status_code=400,
                detail="Uploaded checkpoint is a state_dict, not a callable model",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(trajectory=trajectory, success=True)


@app.post(
    "/api/v2/models/{model_id}/batch-predict",
    response_model=BatchPredictionResponse,
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
)
async def batch_predict(
    model_id: str,
    req: BatchPredictionRequest,
    request: Request,
) -> BatchPredictionResponse:
    """Run batch predictions using an uploaded model."""
    rate_limiter.check(request)
    await require_auth(request)

    if model_id not in _loaded_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    results = []
    for sample in req.samples:
        try:
            resp = await predict(model_id, sample, request)
            results.append(resp)
        except HTTPException:
            results.append(PredictionResponse(trajectory=[], success=False))

    return BatchPredictionResponse(results=results, success=True)


@app.get("/api/v2/models", responses={401: {"model": ErrorResponse}})
async def list_uploaded_models(request: Request) -> dict:
    """List all uploaded model IDs."""
    rate_limiter.check(request)
    await require_auth(request)
    return {"models": list(_loaded_models.keys()), "count": len(_loaded_models)}


# ── Entry point ──────────────────────────────────────────────────────


def main() -> None:
    """Entry point for reactor-twin-api command."""
    import uvicorn

    logger.info("Starting ReactorTwin API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)


__all__ = ["app", "main"]
