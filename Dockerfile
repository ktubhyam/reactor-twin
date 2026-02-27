# ReactorTwin Docker Image
# CPU variant — for GPU, use nvidia/cuda base image

FROM python:3.11-slim AS base

LABEL org.opencontainers.image.source="https://github.com/ktubhyam/reactor-twin"
LABEL org.opencontainers.image.description="Physics-constrained Neural DEs for chemical reactor digital twins"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specification first for layer caching
COPY pyproject.toml ./
COPY src/ ./src/

# Install the package with API, dashboard, and digital twin extras
RUN pip install --no-cache-dir -e ".[api,dashboard,digital_twin,deploy]"

# Copy remaining files
COPY examples/ ./examples/
COPY notebooks/ ./notebooks/

EXPOSE 8000 8501

# Default: run the API server
CMD ["uvicorn", "reactor_twin.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
