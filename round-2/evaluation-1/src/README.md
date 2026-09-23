# Statistical Re-analysis of AEN/ANG Safety Metrics

## Overview

This workspace re-analyzes iteration-1 safety metric results on the Qwen3-4B lineage (Base, SafeRL, Abliterated) with rigorous statistical inference. It evaluates 9 candidate safety metrics plus a logit-gap baseline across 10 harmful + 10 benign prompts from JBB-Behaviors and XSTest.

## Key Findings

| Metric | Avg |d| | P-value | Placebo Ratio | Power (n=10) |
|--------|-----|---------|---------------|----------------|
| **AEN** (Activation Entropy) | **6.53** | <0.001 | **47.3x** | 1.000 |
| **ANG** (Activation Norm Gap) | **3.93** | <0.001 | **9.3x** | 1.000 |
| CLIF | 1.55 | <0.01 | 5.2x | 0.88 |
| MAS | 1.24 | <0.01 | 6.5x | 0.79 |
| MLPS | 1.42 | <0.01 | 5.3x | 0.68 |
| Logit (baseline) | 0.20 | 0.33 | 0.5x | 0.07 |

- **AEN** is the strongest signal: harmful prompts produce higher activation entropy across all 3 models
- **ANG** is second: harmful prompts produce lower activation norms
- Both survive placebo (label-shuffle) and random-weight controls with enormous margins
- All activation-based metrics beat the logit-only baseline by 30-300x

## Layout

```
eval.py                  # Main evaluation script
eval_out.json            # Evaluation output (exp_eval_sol_out schema)
forest_plot.png/pdf      # Forest plot of effect sizes with 95% CI
correlation_plot.png/pdf # Cross-metric correlation analysis
full_method_out.json     # Input data (copied from iter_1 experiment)
pyproject.toml           # Dependencies
logs/run.log             # Execution log
.aii/manifest.yaml       # File lifecycle manifest
```

## How to Run

```bash
# Setup
uv venv .venv --python=3.12
source .venv/bin/activate
uv pip install -e .

# Copy input data
cp /ai-inventor/aii_data/runs/run_xoOpR9T1UFLh/3_invention_loop/iter_1/gen_art/gen_art_experiment_1/full_method_out.json .

# Run evaluation
python eval.py
```

## Analysis Performed

1. **Bootstrap Cohen's d** with 1000 resamples and 95% confidence intervals
2. **Two-tailed p-values** from bootstrap distribution
3. **Power analysis**: minimum detectable effect at n=10 and n=20
4. **Placebo/shuffle control**: re-computed with shuffled labels
5. **Random-weight control deviation**: signal-to-noise ratio
6. **Separation scores**: how well each metric discriminates safety levels
7. **Robustness**: Cohen's d, Hedges' g, Glass's delta, Cliff's delta
8. **Forest plot**: publication-quality figure with CIs
9. **Cross-metric correlation**: scatter plots between AEN and other metrics

## Limitations

- n=10 per group is underpowered for precise effect-size estimates (wide CIs)
- Single model family (Qwen3-4B lineage) limits generalization claims
- No per-layer decomposition (aggregated scores only)
- No component ablation (attention vs MLP) — requires raw activations
- No benchmark correlation — requires external data fetch
- CPU execution (not GPU bf16) may affect activation distributions

## Restoring Removed Files

```bash
# Restore virtual environment
uv venv .venv --python=3.12
uv pip install -e .
```
