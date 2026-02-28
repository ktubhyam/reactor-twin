"""Tests for the ReactorTwin FastAPI server."""

from __future__ import annotations

import json

import pytest

# Guard: skip entire module if fastapi is not installed
pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from reactor_twin.api.server import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


# ── Health check ─────────────────────────────────────────────────────


class TestHealthCheck:
    """Tests for GET /health."""

    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data == {"status": "ok"}


# ── List reactors ───────────────────────────────────────────────────


class TestListReactors:
    """Tests for GET /reactors."""

    def test_reactors_returns_200(self, client: TestClient) -> None:
        response = client.get("/reactors")
        assert response.status_code == 200

    def test_reactors_has_reactors_key(self, client: TestClient) -> None:
        data = client.get("/reactors").json()
        assert "reactors" in data

    def test_reactors_value_is_list(self, client: TestClient) -> None:
        data = client.get("/reactors").json()
        assert isinstance(data["reactors"], list)

    def test_reactors_contains_known_entries(self, client: TestClient) -> None:
        reactors = client.get("/reactors").json()["reactors"]
        assert "exothermic_ab" in reactors
        assert "van_de_vusse" in reactors


# ── Simulate ────────────────────────────────────────────────────────


class TestSimulate:
    """Tests for POST /simulate/{reactor_name}."""

    def test_simulate_exothermic_returns_200(self, client: TestClient) -> None:
        response = client.post("/simulate/exothermic_ab")
        assert response.status_code == 200

    def test_simulate_exothermic_has_required_keys(self, client: TestClient) -> None:
        data = client.post("/simulate/exothermic_ab").json()
        for key in ("time", "states", "labels", "success"):
            assert key in data, f"Missing key: {key}"

    def test_simulate_exothermic_success_flag(self, client: TestClient) -> None:
        data = client.post("/simulate/exothermic_ab").json()
        assert data["success"] is True

    def test_simulate_unknown_reactor_returns_404(self, client: TestClient) -> None:
        response = client.post("/simulate/nonexistent_reactor")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_simulate_custom_parameters(self, client: TestClient) -> None:
        response = client.post(
            "/simulate/exothermic_ab",
            params={"t_end": 5.0, "num_points": 50},
        )
        data = response.json()
        assert data["success"] is True
        assert len(data["time"]) == 50

    def test_simulate_default_num_points(self, client: TestClient) -> None:
        data = client.post("/simulate/exothermic_ab").json()
        assert len(data["time"]) == 100  # default

    def test_simulate_shapes_consistent(self, client: TestClient) -> None:
        num_points = 75
        data = client.post(
            "/simulate/exothermic_ab",
            params={"num_points": num_points},
        ).json()
        assert len(data["time"]) == num_points
        assert len(data["states"]) == num_points
        # Each state row should match the number of labels
        num_labels = len(data["labels"])
        for row in data["states"]:
            assert len(row) == num_labels

    def test_simulate_van_de_vusse(self, client: TestClient) -> None:
        data = client.post("/simulate/van_de_vusse").json()
        assert data["success"] is True
        assert "time" in data


# ── Query parameter validation (422) ─────────────────────────────────


class TestSimulateValidation:
    """Invalid query parameters should return 422."""

    def test_t_end_zero_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/simulate/exothermic_ab",
            params={"t_end": 0.0},
        )
        assert response.status_code == 422

    def test_t_end_negative_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/simulate/exothermic_ab",
            params={"t_end": -5.0},
        )
        assert response.status_code == 422

    def test_t_end_too_large_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/simulate/exothermic_ab",
            params={"t_end": 2000.0},
        )
        assert response.status_code == 422

    def test_num_points_one_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/simulate/exothermic_ab",
            params={"num_points": 1},
        )
        assert response.status_code == 422

    def test_num_points_too_large_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/simulate/exothermic_ab",
            params={"num_points": 20000},
        )
        assert response.status_code == 422


# ── WebSocket ────────────────────────────────────────────────────────


class TestWebSocket:
    """Tests for the /ws/simulate WebSocket endpoint."""

    def test_websocket_connect_and_stream(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/simulate") as ws:
            ws.send_json({
                "reactor_name": "exothermic_ab",
                "t_end": 1.0,
                "num_points": 10,
                "chunk_size": 5,
            })
            messages = []
            while True:
                msg = ws.receive_json()
                messages.append(msg)
                if "status" in msg and msg["status"] == "complete":
                    break
                if "error" in msg:
                    break
            # Should have 2 chunks + 1 completion
            assert len(messages) >= 2
            assert messages[-1]["status"] == "complete"
            assert "labels" in messages[-1]

    def test_websocket_unknown_reactor(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/simulate") as ws:
            ws.send_json({"reactor_name": "nonexistent"})
            msg = ws.receive_json()
            assert "error" in msg

    def test_websocket_invalid_json(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/simulate") as ws:
            ws.send_text("not valid json {{{")
            msg = ws.receive_json()
            assert "error" in msg

    def test_websocket_chunk_sizes(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/simulate") as ws:
            ws.send_json({
                "reactor_name": "exothermic_ab",
                "t_end": 1.0,
                "num_points": 20,
                "chunk_size": 7,
            })
            chunks = []
            while True:
                msg = ws.receive_json()
                if "chunk_index" in msg:
                    chunks.append(msg)
                if "status" in msg or "error" in msg:
                    break
            # 20 points / chunk_size 7 => 3 chunks (7+7+6)
            assert len(chunks) == 3
            assert len(chunks[0]["time"]) == 7
            assert len(chunks[1]["time"]) == 7
            assert len(chunks[2]["time"]) == 6

    def test_websocket_completion_message(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/simulate") as ws:
            ws.send_json({
                "reactor_name": "van_de_vusse",
                "t_end": 2.0,
                "num_points": 10,
                "chunk_size": 100,
            })
            messages = []
            while True:
                msg = ws.receive_json()
                messages.append(msg)
                if "status" in msg or "error" in msg:
                    break
            completion = messages[-1]
            assert completion["status"] == "complete"
            assert isinstance(completion["labels"], list)
