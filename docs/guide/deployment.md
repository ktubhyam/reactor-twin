# Deployment Guide

This guide covers deploying ReactorTwin in production environments.

## Quick Start

### Install from PyPI

```bash
pip install reactor-twin[api]
```

### Start the API Server

```bash
reactor-twin serve --host 0.0.0.0 --port 8000
```

### Train a Model

```bash
reactor-twin train --config config.yaml --output model.pt
```

### Export to ONNX

```bash
reactor-twin export --model model.pt --format onnx --output model.onnx
```

### Launch Dashboard

```bash
reactor-twin dashboard --port 8501
```

---

## Docker Deployment

### CPU Image

```bash
docker build -t reactor-twin:latest .
docker run -p 8000:8000 reactor-twin:latest
```

### GPU Image

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
docker build -f Dockerfile.gpu -t reactor-twin:gpu .
docker run --gpus all -p 8000:8000 reactor-twin:gpu
```

### Docker Compose

```bash
docker compose up -d
```

This starts both the API server (port 8000) and the Streamlit dashboard (port 8501).

---

## Kubernetes with Helm

### Prerequisites

- Kubernetes cluster (1.24+)
- Helm 3.x
- Container image pushed to a registry

### Install

```bash
helm install reactor-twin deploy/helm/reactor-twin \
  --set image.repository=your-registry/reactor-twin \
  --set image.tag=1.0.0
```

### Configure

Edit `deploy/helm/reactor-twin/values.yaml` or pass `--set` flags:

```bash
helm install reactor-twin deploy/helm/reactor-twin \
  --set replicaCount=3 \
  --set autoscaling.maxReplicas=20 \
  --set env.REACTOR_TWIN_JWT_SECRET=your-secret-here
```

### Enable Ingress

```bash
helm install reactor-twin deploy/helm/reactor-twin \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=reactor-twin.example.com \
  --set ingress.hosts[0].paths[0].path=/ \
  --set ingress.hosts[0].paths[0].pathType=Prefix
```

### Upgrade

```bash
helm upgrade reactor-twin deploy/helm/reactor-twin --set image.tag=1.1.0
```

### Uninstall

```bash
helm uninstall reactor-twin
```

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REACTOR_TWIN_JWT_SECRET` | `reactor-twin-dev-secret` | JWT signing secret (change in production!) |
| `REACTOR_TWIN_TOKEN_EXPIRY` | `3600` | Token expiry in seconds |
| `REACTOR_TWIN_RATE_LIMIT` | `60` | Requests per minute per client |

### CLI Commands

| Command | Description |
|---------|-------------|
| `reactor-twin train --config config.yaml` | Train model from YAML config |
| `reactor-twin serve --host 0.0.0.0 --port 8000` | Start API server |
| `reactor-twin export --model model.pt --format onnx` | Export to ONNX |
| `reactor-twin dashboard --port 8501` | Launch Streamlit dashboard |

---

## Monitoring

### Health Endpoint

The API exposes a health check at `GET /health`:

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

Use this for Kubernetes liveness and readiness probes (already configured in the Helm chart).

### Logging

ReactorTwin uses Python's standard `logging` module. Configure log levels via environment variables or programmatically:

```python
import logging
logging.getLogger("reactor_twin").setLevel(logging.DEBUG)
```

---

## Security

### JWT Authentication

All v2 API endpoints require JWT Bearer tokens:

```bash
# Get a token
curl -X POST http://localhost:8000/api/v2/token \
  -H "Content-Type: application/json" \
  -d '{"subject": "my-api-key"}'

# Use the token
curl -X GET http://localhost:8000/api/v2/models \
  -H "Authorization: Bearer <token>"
```

**Production checklist:**
- Set `REACTOR_TWIN_JWT_SECRET` to a strong random value
- Reduce `REACTOR_TWIN_TOKEN_EXPIRY` as appropriate
- Deploy behind TLS (HTTPS)

### Rate Limiting

In-memory sliding window rate limiter is enabled by default (60 requests/minute per client). Configure via `REACTOR_TWIN_RATE_LIMIT`.

### CORS

CORS is configured to allow all origins by default. For production, restrict origins in `src/reactor_twin/api/server.py`.
