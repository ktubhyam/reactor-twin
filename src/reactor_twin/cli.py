"""Unified CLI for ReactorTwin.

Provides commands for training, serving, exporting, and launching the dashboard.
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def cmd_train(args: argparse.Namespace) -> None:
    """Train a Neural ODE model from a YAML config."""
    from reactor_twin.utils.config import load_config

    config = load_config(args.config)
    logger.info(f"Loaded config: {args.config}")

    import numpy as np
    import torch

    torch.manual_seed(config.seed)

    # Create reactor
    from reactor_twin.reactors.systems import create_exothermic_cstr, create_van_de_vusse_cstr

    reactor_factories = {
        "exothermic_ab": create_exothermic_cstr,
        "van_de_vusse": create_van_de_vusse_cstr,
    }

    reactor_type = config.reactor.reactor_type
    if reactor_type not in reactor_factories:
        print(f"Unknown reactor type: {reactor_type}. Available: {list(reactor_factories.keys())}")
        sys.exit(1)

    reactor = reactor_factories[reactor_type]()  # type: ignore[operator]

    # Create model
    from reactor_twin.core.neural_ode import NeuralODE

    model = NeuralODE(
        state_dim=reactor.state_dim,
        hidden_dim=config.neural_de.hidden_dims[0] if config.neural_de.hidden_dims else 64,
        num_layers=len(config.neural_de.hidden_dims),
        solver=config.neural_de.solver,
        atol=config.neural_de.atol,
        rtol=config.neural_de.rtol,
    )

    # Create trainer
    from reactor_twin.training.data_generator import ReactorDataGenerator
    from reactor_twin.training.trainer import Trainer

    gen = ReactorDataGenerator(reactor)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    trainer = Trainer(
        model=model,
        data_generator=gen,
        optimizer=optimizer,
    )

    t_span = (0.0, 10.0)
    t_eval = np.linspace(0.0, 10.0, 50)

    history = trainer.train(
        num_epochs=config.training.n_epochs,
        batch_size=config.training.batch_size,
        t_span=t_span,
        t_eval=t_eval,
    )

    print(f"Training complete. Final loss: {history['train_loss'][-1]:.6f}")

    if args.output:
        model.save(args.output)
        print(f"Model saved to {args.output}")


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI server."""
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required. Install with: pip install reactor-twin[api]")
        sys.exit(1)

    from reactor_twin.api.server import app

    print(f"Starting ReactorTwin API on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_export(args: argparse.Namespace) -> None:
    """Export a model checkpoint to ONNX."""
    if args.format != "onnx":
        print(f"Unsupported export format: {args.format}. Supported: onnx")
        sys.exit(1)

    try:
        from reactor_twin.export.onnx_export import ONNXExporter
    except ImportError:
        print("ONNX export requires: pip install reactor-twin[deploy]")
        sys.exit(1)

    from pathlib import Path

    import torch

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)

    from reactor_twin.core.neural_ode import NeuralODE

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model = NeuralODE(
            state_dim=checkpoint["state_dim"],
            input_dim=checkpoint.get("input_dim", 0),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, NeuralODE):
        model = checkpoint
    else:
        print("Unsupported checkpoint format.")
        sys.exit(1)

    output_path = Path(args.output)
    output_dir = output_path.parent or Path(".")
    ONNXExporter.export(model, output_dir)
    print(f"Model exported to {output_dir}")


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Launch the Streamlit dashboard."""
    import subprocess
    from pathlib import Path

    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    if not dashboard_path.exists():
        print(f"Dashboard not found at {dashboard_path}")
        sys.exit(1)

    cmd = ["streamlit", "run", str(dashboard_path), "--server.port", str(args.port)]
    print(f"Launching dashboard on port {args.port}")
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the reactor-twin CLI."""
    parser = argparse.ArgumentParser(
        prog="reactor-twin",
        description="ReactorTwin: Physics-constrained Neural DEs for chemical reactor digital twins",
    )
    parser.add_argument(
        "--version", action="version", version="reactor-twin 1.0.0"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # train
    train_parser = subparsers.add_parser("train", help="Train a Neural ODE model")
    train_parser.add_argument("--config", required=True, help="Path to YAML config file")
    train_parser.add_argument("--output", default=None, help="Path to save trained model")
    train_parser.set_defaults(func=cmd_train)

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    serve_parser.set_defaults(func=cmd_serve)

    # export
    export_parser = subparsers.add_parser("export", help="Export model to ONNX")
    export_parser.add_argument("--model", required=True, help="Path to model checkpoint")
    export_parser.add_argument("--format", default="onnx", help="Export format (onnx)")
    export_parser.add_argument("--output", default="model.onnx", help="Output path")
    export_parser.set_defaults(func=cmd_export)

    # dashboard
    dash_parser = subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")
    dash_parser.add_argument("--port", type=int, default=8501, help="Dashboard port")
    dash_parser.set_defaults(func=cmd_dashboard)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    logging.basicConfig(level=logging.INFO)
    args.func(args)


__all__ = ["main"]
