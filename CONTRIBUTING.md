# Contributing to ReactorTwin

Thank you for your interest in contributing to ReactorTwin!

## Development Setup

```bash
# Clone repository
git clone https://github.com/ktubhyam/reactor-twin.git
cd reactor-twin

# Install in development mode with all dependencies
pip install -e ".[all]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

ReactorTwin follows strict coding standards:

### Python Style
- **Python 3.10+** - Use modern syntax (`X | Y` unions, match statements)
- **Type hints** - All function signatures must have type hints
- **Docstrings** - Google-style docstrings for all public APIs
- **Import order** - stdlib → third-party → local (enforced by ruff)
- **Logging** - Use `logging` module, never `print()`

### Example
```python
"""Module docstring explaining purpose."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from reactor_twin.core import AbstractNeuralDE

logger = logging.getLogger(__name__)


def my_function(
    x: torch.Tensor,
    param: float = 1.0,
) -> torch.Tensor:
    """Short description of function.

    Args:
        x: Input tensor, shape (batch, dim).
        param: Parameter value.

    Returns:
        Output tensor, shape (batch, dim).
    """
    logger.debug(f"Processing with param={param}")
    return x * param
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=reactor_twin --cov-report=html

# Run specific test file
pytest tests/test_core/test_neural_ode.py

# Run property-based tests
pytest tests/test_integration/test_conservation_laws.py
```

## Code Quality

Before submitting a PR:

```bash
# Format code
ruff format src/

# Lint code
ruff check src/

# Type check
mypy src/

# All checks pass? Good to commit!
```

## Adding New Components

### Adding a New Reactor

1. Create `src/reactor_twin/reactors/my_reactor.py`
2. Inherit from `AbstractReactor`
3. Implement required methods:
   - `_compute_state_dim()`
   - `ode_rhs(t, y, u)`
   - `get_initial_state()`
   - `get_state_labels()`
   - `from_dict(config)`
4. Register with decorator:
   ```python
   @REACTOR_REGISTRY.register("my_reactor")
   class MyReactor(AbstractReactor):
       ...
   ```
5. Add to `reactors/__init__.py`
6. Write tests in `tests/test_reactors/`

### Adding a New Neural DE Variant

1. Create `src/reactor_twin/core/my_neural_de.py`
2. Inherit from `AbstractNeuralDE`
3. Implement required methods:
   - `forward(z0, t_span, controls)`
   - `compute_loss(predictions, targets)`
4. Register with decorator:
   ```python
   @NEURAL_DE_REGISTRY.register("my_neural_de")
   class MyNeuralDE(AbstractNeuralDE):
       ...
   ```
5. Add to `core/__init__.py`
6. Write tests in `tests/test_core/`

### Adding a New Physics Constraint

1. Create `src/reactor_twin/physics/my_constraint.py`
2. Inherit from `AbstractConstraint`
3. Implement required methods:
   - `project(z)` - Hard constraint projection
   - `compute_violation(z)` - Soft constraint penalty
4. Register with decorator:
   ```python
   @CONSTRAINT_REGISTRY.register("my_constraint")
   class MyConstraint(AbstractConstraint):
       ...
   ```
5. Add to `physics/__init__.py`
6. Write tests in `tests/test_physics/`

## Pull Request Process

1. **Fork** the repository
2. **Create branch** from `main`: `git checkout -b feature/my-feature`
3. **Make changes** following code style
4. **Write tests** for new functionality
5. **Update documentation** (docstrings, README, examples)
6. **Run tests and linters** - all must pass
7. **Commit** with conventional commit format:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation only
   - `test:` - Adding tests
   - `refactor:` - Code refactoring
   - `perf:` - Performance improvement
   - `chore:` - Maintenance
8. **Push** to your fork
9. **Open PR** with description of changes

## Questions?

- Open an issue on GitHub
- Email: takarthikeyan25@gmail.com

Thank you for contributing!
