"""WebSocket endpoint for streaming reactor simulation."""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)


async def simulate_ws(websocket: Any, benchmarks: dict[str, Any]) -> None:
    """WebSocket handler for streaming simulation.

    Protocol:
        1. Client connects to ``/ws/simulate``
        2. Client sends JSON config::

            {"reactor_name": "...", "t_end": 10.0,
             "num_points": 100, "chunk_size": 20}

        3. Server streams chunks::

            {"chunk_index": 0, "time": [...], "states": [...]}

        4. Server sends completion::

            {"status": "complete", "labels": [...]}

        5. On error::

            {"error": "message"}  + close

    Args:
        websocket: Starlette/FastAPI WebSocket object.
        benchmarks: Dictionary mapping reactor names to factory functions.
    """
    await websocket.accept()

    try:
        raw = await websocket.receive_text()
        try:
            config = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.send_json({"error": "Invalid JSON"})
            await websocket.close()
            return

        reactor_name = config.get("reactor_name", "")
        t_end = float(config.get("t_end", 10.0))
        num_points = int(config.get("num_points", 100))
        chunk_size = int(config.get("chunk_size", 20))

        if reactor_name not in benchmarks:
            available = list(benchmarks.keys())
            await websocket.send_json(
                {"error": f"Unknown reactor: {reactor_name}. Available: {available}"}
            )
            await websocket.close()
            return

        # Run simulation
        reactor = benchmarks[reactor_name]()
        y0 = reactor.get_initial_state()
        t_eval = np.linspace(0, t_end, num_points)

        sol = solve_ivp(
            reactor.ode_rhs,
            [0, t_end],
            y0,
            t_eval=t_eval,
            method="RK45",
        )

        if not sol.success:
            await websocket.send_json({"error": f"Solver failed: {sol.message}"})
            await websocket.close()
            return

        # Stream chunks
        times = sol.t.tolist()
        states = sol.y.T.tolist()

        for chunk_index, start in enumerate(range(0, num_points, chunk_size)):
            end = min(start + chunk_size, num_points)
            await websocket.send_json(
                {
                    "chunk_index": chunk_index,
                    "time": times[start:end],
                    "states": states[start:end],
                }
            )

        # Completion message
        await websocket.send_json(
            {
                "status": "complete",
                "labels": reactor.get_state_labels(),
            }
        )

    except Exception as exc:
        logger.exception("WebSocket error")
        with contextlib.suppress(Exception):
            await websocket.send_json({"error": str(exc)})
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()


__all__ = ["simulate_ws"]
