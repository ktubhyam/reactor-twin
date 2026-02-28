"""Load testing script for the ReactorTwin API.

Uses concurrent requests to test API performance under load.
Does not require locust/k6 — uses Python's built-in ``concurrent.futures``.

Usage:
    # Start the API server first:
    reactor-twin-api
    # Then in another terminal:
    python scripts/loadtest.py --url http://localhost:8000 --concurrency 10 --requests 100
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def _make_request(url: str, method: str = "GET", data: dict | None = None) -> tuple[int, float]:
    """Make a single HTTP request and return (status_code, latency_ms)."""
    start = time.perf_counter()
    req = Request(url, method=method)
    req.add_header("Content-Type", "application/json")

    if data is not None:
        req.data = json.dumps(data).encode()
        req.method = "POST"

    try:
        with urlopen(req, timeout=30) as resp:
            resp.read()
            status = resp.status
    except HTTPError as e:
        status = e.code

    elapsed_ms = (time.perf_counter() - start) * 1000
    return status, elapsed_ms


def run_load_test(
    base_url: str,
    concurrency: int = 10,
    total_requests: int = 100,
) -> dict:
    """Run load test against the API.

    Tests:
    1. Health check endpoint (GET /health)
    2. Reactor list endpoint (GET /reactors)
    3. Simulation endpoint (POST /simulate/exothermic_ab)

    Args:
        base_url: API base URL.
        concurrency: Number of concurrent workers.
        total_requests: Total number of requests per endpoint.

    Returns:
        Dict with latency statistics per endpoint.
    """
    endpoints = {
        "GET /health": {"url": f"{base_url}/health", "method": "GET"},
        "GET /reactors": {"url": f"{base_url}/reactors", "method": "GET"},
        "POST /simulate": {
            "url": f"{base_url}/simulate/exothermic_ab?t_end=5&num_points=50",
            "method": "POST",
        },
    }

    results = {}

    for name, config in endpoints.items():
        print(f"\n--- {name} ({total_requests} requests, {concurrency} workers) ---")
        latencies = []
        errors = 0

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for _ in range(total_requests):
                futures.append(
                    executor.submit(_make_request, config["url"], config["method"])
                )

            for future in as_completed(futures):
                status, latency = future.result()
                latencies.append(latency)
                if status >= 400:
                    errors += 1

        # Statistics
        latencies.sort()
        stats = {
            "total_requests": total_requests,
            "errors": errors,
            "error_rate": errors / total_requests * 100,
            "min_ms": round(min(latencies), 1),
            "max_ms": round(max(latencies), 1),
            "mean_ms": round(statistics.mean(latencies), 1),
            "median_ms": round(statistics.median(latencies), 1),
            "p95_ms": round(latencies[int(0.95 * len(latencies))], 1),
            "p99_ms": round(latencies[int(0.99 * len(latencies))], 1),
            "throughput_rps": round(total_requests / (sum(latencies) / 1000 / concurrency), 1),
        }

        results[name] = stats

        print(f"  Errors:     {stats['errors']}/{total_requests} ({stats['error_rate']:.1f}%)")
        print(f"  Min:        {stats['min_ms']:.1f} ms")
        print(f"  Mean:       {stats['mean_ms']:.1f} ms")
        print(f"  Median:     {stats['median_ms']:.1f} ms")
        print(f"  P95:        {stats['p95_ms']:.1f} ms")
        print(f"  P99:        {stats['p99_ms']:.1f} ms")
        print(f"  Max:        {stats['max_ms']:.1f} ms")
        print(f"  Throughput: {stats['throughput_rps']:.1f} req/s")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="ReactorTwin API Load Test")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent workers")
    parser.add_argument("--requests", type=int, default=100, help="Total requests per endpoint")
    args = parser.parse_args()

    print("ReactorTwin API Load Test")
    print(f"URL: {args.url}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Requests per endpoint: {args.requests}")

    results = run_load_test(args.url, args.concurrency, args.requests)

    print("\n\n=== SUMMARY ===")
    for name, stats in results.items():
        status = "PASS" if stats["error_rate"] < 5 else "FAIL"
        print(f"  [{status}] {name}: mean={stats['mean_ms']:.0f}ms, "
              f"p95={stats['p95_ms']:.0f}ms, {stats['throughput_rps']:.0f} rps")


if __name__ == "__main__":
    main()
