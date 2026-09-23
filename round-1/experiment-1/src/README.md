# Safety Metric Evaluation on Qwen3-4B Lineage

## Overview

This experiment evaluates **9 candidate safety metrics** (plus 1 logit-gap baseline) on a Qwen3-4B model lineage to identify internal activation signals that distinguish safe, base, and abliterated models using only a few prompts.

## Models Tested

| Model | Label | Source |
|-------|-------|--------|
| Qwen/Qwen3-4B-Base | Base | Official base model |
| Qwen/Qwen3-4B-SafeRL | SafeRL | Official safety RL model |
| huihui-ai/Huihui-Qwen3-4B-abliterated-v2 | Abliterated | Community abliterated variant |

## Metrics

### Activation-Based (read hidden states)
1. **ATC** — Activation Trajectory Consistency: cosine similarity between harmful/benign activation trajectories
2. **CLIF** — Cross-Layer Information Flow: stability of activation norm ratios across layers
3. **MAS** — MLP Activation Sparsity: fraction of near-zero activations (threshold-based)
4. **ANG** — Activation Norm Gap: mean L2 norm of activations per prompt
5. **AEN** — Activation Entropy: Shannon entropy of activation distribution
6. **LWCD** — Layer-Wise Cosine Distance: cosine distance between mean harmful/benign vectors
7. **MLPS** — MLP Output Sparsity: fraction of activations within 0.1σ of zero
8. **RSD** — Residual Stream Direction: cosine similarity between consecutive layer activations

### Weight-Based
9. **WSSS** — Weight Spectral Signature: spectral norm of MLP up_proj weights

### Logit-Only Baseline
10. **Logit** — Top-2 logit margin

## Layout

```
method.py           — Main experiment script (all metrics + analysis)
method_out.json     — Results (32K, schema-valid)
logs/full_run.log   — Full execution log
.aii/manifest.yaml  — File retention decisions
```

## How to Run

```bash
# Create environment
uv venv .venv --python=3.12
source .venv/bin/activate
uv pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install transformers datasets numpy scipy loguru

# Run experiment (~10 min on CPU)
python method.py
```

## Key Results

Top metrics by Cohen's d (harmful vs benign separation):
- **AEN**: avg|d| = 6.53 (highest separation)
- **ANG**: avg|d| = 3.93
- **MLPS**: avg|d| = 1.42
- **CLIF**: avg|d| = 1.55
- **MAS**: avg|d| = 1.24

All activation-based metrics beat the logit-only baseline (avg|d| = 0.20).

## Restoring Removed Files

```bash
# Restore .venv
uv venv .venv --python=3.12
uv pip install --python=.venv/bin/python torch --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install --python=.venv/bin/python transformers datasets numpy scipy loguru

# Restore __pycache__ (auto-generated on next run)
python method.py
```
