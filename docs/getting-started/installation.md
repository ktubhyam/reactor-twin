# Installation

## Basic Installation

```bash
pip install reactor-twin
```

## Development Installation

```bash
git clone https://github.com/ktubhyam/reactor-twin.git
cd reactor-twin
pip install -e ".[dev]"
```

## Optional Dependencies

| Extra | Description | Install |
|-------|-------------|---------|
| `dashboard` | Streamlit dashboard | `pip install reactor-twin[dashboard]` |
| `digital_twin` | Fault detection (scikit-learn) | `pip install reactor-twin[digital_twin]` |
| `api` | REST API server | `pip install reactor-twin[api]` |
| `sde` | Neural SDE support | `pip install reactor-twin[sde]` |
| `cde` | Neural CDE support | `pip install reactor-twin[cde]` |
| `all` | Everything | `pip install reactor-twin[all]` |

## Requirements

- Python >= 3.10
- PyTorch >= 2.1
- NumPy >= 1.24, < 2.0
