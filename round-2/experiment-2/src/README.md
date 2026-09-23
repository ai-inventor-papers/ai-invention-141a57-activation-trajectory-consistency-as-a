# AEN/ANG Safety Metrics Across Model Families

Tests whether internal activation patterns (AEN, ANG) predict AI safety across 4 model families using just 20 prompts per model.

## Layout

| File | Description |
|------|-------------|
| `method.py` | Main experiment script |
| `pyproject.toml` | Dependencies (pinned) |
| `method_out.json` | Full results (360 examples) |
| `full_method_out.json` | Copy of full results |
| `mini_method_out.json` | Truncated results (3 examples) |
| `preview_method_out.json` | Truncated + string-limited results |
| `results/` | Output directory |
| `logs/` | Experiment logs |

## How to Run

```bash
uv venv .venv --python=3.12 && uv sync
uv run method.py
```

## Restoring Removed Files

```bash
# Restore .venv (regenerable)
uv venv .venv --python=3.12 && uv sync
```
