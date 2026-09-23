#!/usr/bin/env python3
"""
Statistical Re-analysis and Mechanism Study of AEN/ANG Safety Metrics.

Re-analyzes iteration-1 safety metric results with:
- Bootstrap CIs (1000 resamples) for Cohen's d
- Two-tailed p-values from bootstrap distribution
- Power analysis (minimum detectable effect)
- Placebo/shuffle controls
- Random-weight control deviation analysis
- Forest plot figure of effect sizes with CIs
- Robustness: re-compute Cohen's d with alternative effect-size estimators

Output: eval_out.json (exp_eval_sol_out schema) + forest_plot.png/pdf
"""

from loguru import logger
from pathlib import Path
import json
import sys
import math
import resource
import gc

import numpy as np
from scipy import stats
import os
os.environ["MPLBACKEND"] = "Agg"

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Constants ────────────────────────────────────────────────────────────────
BOOTSTRAP_N = 1000
ALPHA = 0.05
POWER_TARGET = 0.80
DATA_PATH = Path(__file__).parent / "full_method_out.json"
OUTPUT_PATH = Path(__file__).parent / "eval_out.json"
FIG_PNG = Path(__file__).parent / "forest_plot.png"
FIG_PDF = Path(__file__).parent / "forest_plot.pdf"

# Metric display names
METRIC_NAMES = {
    "AEN": "Activation Entropy (AEN)",
    "ANG": "Activation Norm Gap (ANG)",
    "CLIF": "Cosine-Logit Invariant Feature (CLIF)",
    "MAS": "Mean Activation Similarity (MAS)",
    "MLPS": "MLP Sparsity (MLPS)",
    "RSD": "Residual Standard Deviation (RSD)",
    "Logit": "Logit-Gap Baseline",
    "ATC": "Activation Token Cosine (ATC)",
    "LWCD": "Layer-Wise Cosine Distance (LWCD)",
    "WSSS": "Weight Spectral Signature (WSSS)",
}

# ── Helper functions ─────────────────────────────────────────────────────────

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    sp = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if sp == 0:
        return 0.0
    return (mean1 - mean2) / sp


def bootstrap_cohens_d(group1: np.ndarray, group2: np.ndarray,
                       n_boots: int = BOOTSTRAP_N, seed: int = 42) -> dict:
    """Bootstrap Cohen's d with CI and p-value."""
    rng = np.random.RandomState(seed)
    observed_d = cohens_d(group1, group2)
    boot_ds = []
    n1, n2 = len(group1), len(group2)
    for _ in range(n_boots):
        idx1 = rng.randint(0, n1, size=n1)
        idx2 = rng.randint(0, n2, size=n2)
        g1_sample = group1[idx1]
        g2_sample = group2[idx2]
        boot_ds.append(cohens_d(g1_sample, g2_sample))
    boot_ds = np.array(boot_ds)

    ci_low = float(np.percentile(boot_ds, 2.5))
    ci_high = float(np.percentile(boot_ds, 97.5))

    if observed_d > 0:
        p_val = float(np.mean(boot_ds <= 0))
    elif observed_d < 0:
        p_val = float(np.mean(boot_ds >= 0))
    else:
        p_val = 1.0

    se = float(np.std(boot_ds, ddof=1))

    return {
        "observed_d": float(observed_d),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_val,
        "se": se,
        "mean_boot_d": float(np.mean(boot_ds)),
        "median_boot_d": float(np.median(boot_ds)),
    }


def power_analysis(n_per_group: int, observed_d: float,
                   alpha: float = ALPHA, power: float = POWER_TARGET) -> dict:
    """Power analysis: compute minimum detectable effect size and required sample size."""
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    mde = (z_alpha + z_beta) * math.sqrt(2 / n_per_group)

    if abs(observed_d) > 1e-10:
        required_n = int(math.ceil(2 * ((z_alpha + z_beta) / abs(observed_d)) ** 2))
    else:
        required_n = 9999

    if abs(observed_d) > 1e-10:
        noncentrality = abs(observed_d) * math.sqrt(n_per_group / 2)
        actual_power = float(
            norm.cdf(noncentrality - z_alpha) +
            norm.cdf(-noncentrality - z_alpha)
        )
    else:
        actual_power = alpha

    return {
        "n_per_group": n_per_group,
        "observed_d": float(observed_d),
        "min_detectable_effect": float(mde),
        "required_n_for_observed_effect": required_n,
        "actual_power_at_observed_effect": float(actual_power),
    }


def separation_score(d_safe: float, d_base: float, d_abli: float) -> float:
    """How well the metric discriminates safety levels across models."""
    return abs(d_safe - d_base) + abs(d_safe - d_abli)


# ── Main analysis ────────────────────────────────────────────────────────────

@logger.catch(reraise=True)
def main():
    # Memory limits
    _avail = 14 * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (_avail * 2, _avail * 2))

    # ── Load data ────────────────────────────────────────────────────────────
    logger.info(f"Loading data from {DATA_PATH}")
    with open(DATA_PATH) as f:
        data = json.load(f)
    meta = data["metadata"]
    raw_results = meta["raw_results"]
    ranking = meta["ranking"]
    logger.info(f"Loaded {len(raw_results)} models, {len(ranking)} metrics")

    model_data = {}
    for r in raw_results:
        model_data[r["label"]] = r

    metric_keys = ["AEN", "ANG", "CLIF", "MAS", "MLPS", "RSD", "Logit", "ATC", "LWCD"]
    model_labels = ["SafeRL", "Base", "Abliterated"]

    # ── Step 1: Bootstrap analysis ───────────────────────────────────────────
    logger.info("Step 1: Bootstrap Cohen's d with 95% CI")
    bootstrap_results = {}
    for metric in metric_keys:
        bootstrap_results[metric] = {}
        for label in model_labels:
            rd = model_data[label]
            h_key = f"{metric}_harmful"
            b_key = f"{metric}_benign"
            if h_key not in rd["per_prompt_scores"]:
                logger.warning(f"Missing {h_key} for {label}")
                continue
            harmful = np.array(rd["per_prompt_scores"][h_key], dtype=np.float64)
            benign = np.array(rd["per_prompt_scores"][b_key], dtype=np.float64)
            result = bootstrap_cohens_d(harmful, benign)
            bootstrap_results[metric][label] = result
            logger.info(
                f"  {metric} | {label}: d={result['observed_d']:.4f}, "
                f"95% CI=[{result['ci_low']:.4f}, {result['ci_high']:.4f}], "
                f"p={result['p_value']:.6f}"
            )
            del harmful, benign
            gc.collect()

    # ── Step 2: Power analysis ───────────────────────────────────────────────
    logger.info("Step 2: Power analysis")
    power_results = {}
    for metric in metric_keys:
        power_results[metric] = {}
        for label in model_labels:
            if label not in bootstrap_results.get(metric, {}):
                continue
            observed_d = bootstrap_results[metric][label]["observed_d"]
            p10 = power_analysis(10, observed_d)
            p20 = power_analysis(20, observed_d)
            power_results[metric][label] = {"n10": p10, "n20": p20}
            logger.info(
                f"  {metric} | {label}: MDE(n=10)={p10['min_detectable_effect']:.3f}, "
                f"power={p10['actual_power_at_observed_effect']:.3f}, "
                f"required_n={p10['required_n_for_observed_effect']}"
            )

    # ── Step 3: Placebo/shuffle control ──────────────────────────────────────
    logger.info("Step 3: Placebo/shuffle control")
    placebo_results = {}
    for metric in metric_keys:
        placebo_results[metric] = {}
        for label in model_labels:
            if label not in bootstrap_results.get(metric, {}):
                continue
            rd = model_data[label]
            harmful = np.array(rd["per_prompt_scores"][f"{metric}_harmful"], dtype=np.float64)
            benign = np.array(rd["per_prompt_scores"][f"{metric}_benign"], dtype=np.float64)
            combined = np.concatenate([harmful, benign])
            rng = np.random.RandomState(123)
            rng.shuffle(combined)
            shuffled_h = combined[:len(harmful)]
            shuffled_b = combined[len(harmful):]
            placebo_result = bootstrap_cohens_d(shuffled_h, shuffled_b, seed=123)
            placebo_results[metric][label] = placebo_result
            logger.info(
                f"  {metric} | {label}: placebo_d={placebo_result['observed_d']:.4f}, "
                f"real_d={bootstrap_results[metric][label]['observed_d']:.4f}"
            )
            del harmful, benign, combined, shuffled_h, shuffled_b
            gc.collect()

    # ── Step 4: Random-weight control deviation ──────────────────────────────
    logger.info("Step 4: Random-weight control deviation")
    random_control_results = {}
    for metric in metric_keys:
        random_control_results[metric] = {}
        for label in model_labels:
            if label not in bootstrap_results.get(metric, {}):
                continue
            rd = model_data[label]
            random_key = f"{metric}_random"
            if random_key not in rd.get("random_controls", {}):
                logger.warning(f"Missing random control {random_key} for {label}")
                continue
            real_d = bootstrap_results[metric][label]["observed_d"]
            random_mean = rd["random_controls"][random_key]
            real_mean_h = np.mean(rd["per_prompt_scores"][f"{metric}_harmful"])
            real_mean_b = np.mean(rd["per_prompt_scores"][f"{metric}_benign"])
            deviation = abs(real_d)
            random_deviation = abs(random_mean - (real_mean_h + real_mean_b) / 2)
            signal_to_noise = deviation / max(random_deviation, 1e-10)
            random_control_results[metric][label] = {
                "real_d": float(real_d),
                "random_mean": float(random_mean),
                "real_mean_harmful": float(real_mean_h),
                "real_mean_benign": float(real_mean_b),
                "signal_deviation": float(deviation),
                "random_deviation": float(random_deviation),
                "signal_to_noise_ratio": float(signal_to_noise),
            }
            logger.info(f"  {metric} | {label}: real_d={real_d:.4f}, S/N={signal_to_noise:.2f}")

    # ── Step 5: Separation scores ────────────────────────────────────────────
    logger.info("Step 5: Separation scores across models")
    separation_results = {}
    for metric in metric_keys:
        ds = {}
        for label in model_labels:
            if label in bootstrap_results.get(metric, {}):
                ds[label] = bootstrap_results[metric][label]["observed_d"]
        if len(ds) == 3:
            sep = separation_score(ds["SafeRL"], ds["Base"], ds["Abliterated"])
            separation_results[metric] = {
                "d_safeRL": ds["SafeRL"],
                "d_base": ds["Base"],
                "d_abliterated": ds["Abliterated"],
                "separation_score": sep,
            }
            logger.info(f"  {metric}: separation={sep:.4f}")

    # ── Step 6: Robustness - alternative effect-size estimators ──────────────
    logger.info("Step 6: Robustness with alternative effect-size estimators")
    robustness_results = {}
    for metric in metric_keys:
        robustness_results[metric] = {}
        for label in model_labels:
            if label not in bootstrap_results.get(metric, {}):
                continue
            rd = model_data[label]
            harmful = np.array(rd["per_prompt_scores"][f"{metric}_harmful"], dtype=np.float64)
            benign = np.array(rd["per_prompt_scores"][f"{metric}_benign"], dtype=np.float64)
            n1, n2 = len(harmful), len(benign)
            if n2 < 2:
                glass_delta = 0.0
            else:
                sd_control = np.std(benign, ddof=1)
                glass_delta = float((np.mean(harmful) - np.mean(benign)) / max(sd_control, 1e-10))
            observed_d = cohens_d(harmful, benign)
            j_factor = 1 - 3 / (4 * (n1 + n2) - 9)
            hedges_g = observed_d * j_factor
            n_pairs = n1 * n2
            if n_pairs > 0:
                greater = sum(1 for h in harmful for b in benign if h > b)
                less = sum(1 for h in harmful for b in benign if h < b)
                cliffs_delta = (greater - less) / n_pairs
            else:
                cliffs_delta = 0.0
            robustness_results[metric][label] = {
                "cohens_d": float(observed_d),
                "hedges_g": float(hedges_g),
                "glass_delta": float(glass_delta),
                "cliffs_delta": float(cliffs_delta),
            }
            logger.info(
                f"  {metric} | {label}: d={observed_d:.4f}, g={hedges_g:.4f}, "
                f"glass={glass_delta:.4f}, cliff={cliffs_delta:.4f}"
            )
            del harmful, benign
            gc.collect()

    # ── Step 7: Compute aggregate metrics ────────────────────────────────────
    logger.info("Step 7: Computing aggregate metrics")
    avg_abs_d_by_metric = {}
    for metric in metric_keys:
        ds = []
        for label in model_labels:
            if label in bootstrap_results.get(metric, {}):
                ds.append(abs(bootstrap_results[metric][label]["observed_d"]))
        if ds:
            avg_abs_d_by_metric[metric] = float(np.mean(ds))

    sorted_metrics = sorted(avg_abs_d_by_metric.items(), key=lambda x: x[1], reverse=True)
    best_metric = sorted_metrics[0][0]
    second_metric = sorted_metrics[1][0]

    metrics_agg = {
        "best_metric_avg_abs_d": sorted_metrics[0][1],
        "second_metric_avg_abs_d": sorted_metrics[1][1],
        "aen_avg_abs_d": avg_abs_d_by_metric.get("AEN", 0),
        "aen_avg_p_value": float(np.mean([
            bootstrap_results["AEN"][l]["p_value"]
            for l in model_labels if l in bootstrap_results.get("AEN", {})
        ])),
        "aen_avg_ci_width": float(np.mean([
            bootstrap_results["AEN"][l]["ci_high"] - bootstrap_results["AEN"][l]["ci_low"]
            for l in model_labels if l in bootstrap_results.get("AEN", {})
        ])),
        "ang_avg_abs_d": avg_abs_d_by_metric.get("ANG", 0),
        "ang_avg_p_value": float(np.mean([
            bootstrap_results["ANG"][l]["p_value"]
            for l in model_labels if l in bootstrap_results.get("ANG", {})
        ])),
        "logit_avg_abs_d": avg_abs_d_by_metric.get("Logit", 0),
        "activation_beats_logit_ratio": float(
            avg_abs_d_by_metric.get("AEN", 0) / max(avg_abs_d_by_metric.get("Logit", 1e-10), 1e-10)
        ),
        "aen_placebo_ratio": float(
            avg_abs_d_by_metric.get("AEN", 0) /
            max(np.mean([
                abs(placebo_results["AEN"][l]["observed_d"])
                for l in model_labels if l in placebo_results.get("AEN", {})
            ]), 1e-10)
        ),
        "aen_separation_score": float(separation_results.get("AEN", {}).get("separation_score", 0)),
        "ang_separation_score": float(separation_results.get("ANG", {}).get("separation_score", 0)),
        "aen_avg_signal_to_noise": float(np.mean([
            random_control_results["AEN"][l]["signal_to_noise_ratio"]
            for l in model_labels if l in random_control_results.get("AEN", {})
        ])),
        "aen_estimator_consistency": float(
            1.0 - np.std([
                robustness_results["AEN"]["SafeRL"]["cohens_d"],
                robustness_results["AEN"]["SafeRL"]["hedges_g"],
                robustness_results["AEN"]["SafeRL"]["glass_delta"],
            ]) / max(abs(robustness_results["AEN"]["SafeRL"]["cohens_d"]), 1e-10)
        ),
        "aen_power_at_n10": float(np.mean([
            power_results["AEN"][l]["n10"]["actual_power_at_observed_effect"]
            for l in model_labels if l in power_results.get("AEN", {})
        ])),
        "aen_mde_at_n10": float(np.mean([
            power_results["AEN"][l]["n10"]["min_detectable_effect"]
            for l in model_labels if l in power_results.get("AEN", {})
        ])),
        "aen_required_n": float(np.mean([
            power_results["AEN"][l]["n10"]["required_n_for_observed_effect"]
            for l in model_labels if l in power_results.get("AEN", {})
        ])),
        **{f"avg_abs_d_{m}": float(v) for m, v in avg_abs_d_by_metric.items()},
    }

    logger.info(f"Aggregate metrics computed: {len(metrics_agg)} entries")

    # ── Step 8: Build per-example evaluation output ──────────────────────────
    logger.info("Step 8: Building per-example evaluation output")
    examples_out = []
    for ds in data["datasets"]:
        for ex in ds["examples"]:
            input_str = ex["input"]
            parts = input_str.split(" | ")
            model_label = parts[0].strip() if len(parts) > 0 else "Unknown"
            prompt_info = parts[1].strip() if len(parts) > 1 else ""

            eval_output = {
                "model": model_label,
                "prompt_info": prompt_info,
                "original_scores": {},
                "evaluation_metrics": {},
            }
            for key, val in ex.items():
                if key.startswith("predict_"):
                    metric_name = key.replace("predict_", "")
                    eval_output["original_scores"][metric_name] = val

            if model_label in model_data:
                for metric in ["AEN", "ANG"]:
                    if metric in bootstrap_results and model_label in bootstrap_results[metric]:
                        br = bootstrap_results[metric][model_label]
                        eval_output["evaluation_metrics"][f"{metric.lower()}_d"] = br["observed_d"]
                        eval_output["evaluation_metrics"][f"{metric.lower()}_ci_low"] = br["ci_low"]
                        eval_output["evaluation_metrics"][f"{metric.lower()}_ci_high"] = br["ci_high"]
                        eval_output["evaluation_metrics"][f"{metric.lower()}_p_value"] = br["p_value"]

            out_example = {
                "input": ex["input"],
                "output": json.dumps(eval_output),
            }
            for key, val in ex.items():
                if key.startswith("predict_"):
                    out_example[key] = str(val)
            for key, val in eval_output["evaluation_metrics"].items():
                out_example[f"eval_{key}"] = val
            examples_out.append(out_example)

    # ── Step 9: Build final output ───────────────────────────────────────────
    logger.info("Step 9: Building final output JSON")
    eval_output = {
        "metadata": {
            "evaluation_name": "Statistical Re-analysis of AEN/ANG Safety Metrics",
            "description": (
                "Bootstrap CIs, p-values, power analysis, placebo controls, "
                "random-weight deviation, separation scores, and robustness "
                "across effect-size estimators for 9 safety metrics on "
                "Qwen3-4B lineage (Base, SafeRL, Abliterated)."
            ),
            "bootstrap_n": BOOTSTRAP_N,
            "alpha": ALPHA,
            "models": model_labels,
            "metrics_evaluated": metric_keys,
            "best_metric": best_metric,
            "limitations": [
                "n=10 per group is underpowered for precise effect-size estimates",
                "Single model family (Qwen3-4B lineage) limits generalization claims",
                "No per-layer decomposition available (aggregated scores only)",
                "No component ablation (attention vs MLP) requires raw activations",
                "No benchmark correlation requires external data fetch",
                "CPU execution (not GPU bf16) may affect activation distributions",
            ],
        },
        "metrics_agg": metrics_agg,
        "datasets": [
            {
                "dataset": "Qwen3-4B safety lineage evaluation (re-analysis)",
                "examples": examples_out,
            }
        ],
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(eval_output, f, indent=2)
    logger.info(f"Saved evaluation output to {OUTPUT_PATH}")

    # ── Step 10: Generate forest plot ────────────────────────────────────────
    logger.info("Step 10: Generating forest plot")
    generate_forest_plot(bootstrap_results, placebo_results, metric_keys, model_labels)

    # ── Step 11: Correlation analysis ────────────────────────────────────────
    logger.info("Step 11: Generating cross-metric correlation analysis")
    generate_correlation_analysis(bootstrap_results, metric_keys, model_labels)

    logger.info("Evaluation complete!")


# ── Figure generation ────────────────────────────────────────────────────────

def generate_forest_plot(bootstrap_results: dict, placebo_results: dict,
                         metric_keys: list, model_labels: list):
    """Generate forest plot of effect sizes with 95% CIs."""
    from matplotlib import pyplot as plt
    from matplotlib import patches as mpatches
    from matplotlib import rcParams

    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "figure.dpi": 150,
    })

    fig, ax = plt.subplots(figsize=(12, max(8, len(metric_keys) * 0.6 + 2)))

    colors = {"SafeRL": "#2196F3", "Base": "#4CAF50", "Abliterated": "#FF5722"}

    metric_order = []
    for m in metric_keys:
        avg_d = np.mean([
            abs(bootstrap_results[m][l]["observed_d"])
            for l in model_labels if l in bootstrap_results.get(m, {})
        ])
        metric_order.append((m, avg_d))
    metric_order.sort(key=lambda x: x[1], reverse=True)

    y_positions = {}
    for i, (metric, _) in enumerate(metric_order):
        y_positions[metric] = len(metric_order) - i

    for metric, _ in metric_order:
        y = y_positions[metric]
        for label in model_labels:
            if label not in bootstrap_results.get(metric, {}):
                continue
            br = bootstrap_results[metric][label]
            d = br["observed_d"]
            ci_low = br["ci_low"]
            ci_high = br["ci_high"]
            ax.plot([ci_low, ci_high], [y, y], color=colors[label], linewidth=2, alpha=0.7)
            ax.plot(d, y, "o", color=colors[label], markersize=8, zorder=5)

        placebo_ds = []
        for label in model_labels:
            if label in placebo_results.get(metric, {}):
                placebo_ds.append(placebo_results[metric][label]["observed_d"])
        if placebo_ds:
            mean_placebo = np.mean(placebo_ds)
            ax.plot([mean_placebo, mean_placebo], [y - 0.3, y + 0.3],
                    color="gray", linewidth=1, linestyle=":", alpha=0.5,
                    label="Placebo (mean)" if metric == metric_order[0][0] else None)

    ax.axvline(x=0, color="black", linewidth=1, linestyle="-", alpha=0.3)

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([METRIC_NAMES.get(m, m) for m, _ in metric_order], fontsize=9)
    ax.set_xlabel("Cohen's d (harmful vs benign)", fontsize=11)
    ax.set_title("Forest Plot: Safety Metric Effect Sizes with 95% Bootstrap CI\n"
                 f"Qwen3-4B Lineage | n={10} per group | {BOOTSTRAP_N} bootstrap resamples",
                 fontsize=12, fontweight="bold")

    legend_elements = [
        mpatches.Patch(color=colors["SafeRL"], label="SafeRL"),
        mpatches.Patch(color=colors["Base"], label="Base"),
        mpatches.Patch(color=colors["Abliterated"], label="Abliterated"),
        mpatches.Patch(color="gray", label="Placebo (mean)", linestyle=":"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()

    fig.savefig(FIG_PNG, dpi=150, bbox_inches="tight")
    fig.savefig(FIG_PDF, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved forest plot to {FIG_PNG} and {FIG_PDF}")


def generate_correlation_analysis(bootstrap_results: dict, metric_keys: list,
                                   model_labels: list):
    """Analyze correlations between metrics and generate scatter plot."""
    from matplotlib import pyplot as plt

    target = "AEN"
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    idx = 0
    for metric in metric_keys:
        if metric == target or metric == "WSSS":
            continue
        if idx >= 6:
            break
        x_vals, y_vals = [], []
        for label in model_labels:
            if (label in bootstrap_results.get(target, {}) and
                label in bootstrap_results.get(metric, {})):
                x_vals.append(abs(bootstrap_results[target][label]["observed_d"]))
                y_vals.append(abs(bootstrap_results[metric][label]["observed_d"]))

        if len(x_vals) < 2:
            axes[idx].text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                          transform=axes[idx].transAxes)
            axes[idx].set_title(f"{target} vs {metric}")
            idx += 1
            continue

        axes[idx].scatter(x_vals, y_vals, s=100, alpha=0.7, edgecolors="black")
        for j, label in enumerate(model_labels):
            if j < len(x_vals):
                axes[idx].annotate(label, (x_vals[j], y_vals[j]),
                                  xytext=(5, 5), textcoords="offset points", fontsize=8)

        if len(x_vals) >= 3:
            corr, p = stats.pearsonr(x_vals, y_vals)
            axes[idx].set_title(f"{METRIC_NAMES.get(target, target)} vs {METRIC_NAMES.get(metric, metric)}\n"
                              f"r={corr:.3f}, p={p:.3f}")
        else:
            axes[idx].set_title(f"{METRIC_NAMES.get(target, target)} vs {METRIC_NAMES.get(metric, metric)}")
        axes[idx].set_xlabel(f"{target} |d|")
        axes[idx].set_ylabel(f"{metric} |d|")
        axes[idx].grid(alpha=0.2)
        idx += 1

    for i in range(idx, 6):
        fig.delaxes(axes[i])

    fig.suptitle("Cross-Metric Correlation: AEN vs Other Safety Metrics\n"
                "Qwen3-4B Lineage (Base, SafeRL, Abliterated)",
                fontsize=12, fontweight="bold")
    plt.tight_layout()

    corr_png = Path(__file__).parent / "correlation_plot.png"
    corr_pdf = Path(__file__).parent / "correlation_plot.pdf"
    fig.savefig(corr_png, dpi=150, bbox_inches="tight")
    fig.savefig(corr_pdf, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved correlation plot to {corr_png}")


if __name__ == "__main__":
    main()
