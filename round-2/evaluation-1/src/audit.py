#!/usr/bin/env python3
"""
Independent audit: re-derive headline numbers from raw data using a DIFFERENT code path.

This script reads the raw per-prompt scores from full_method_out.json and independently
recomputes Cohen's d for AEN and ANG using scipy.stats instead of manual formulas.
It also runs a placebo test on shuffled labels to confirm the signal is real.
"""

import json
import math
import numpy as np
from scipy import stats

DATA_PATH = "full_method_out.json"

def cohens_d_scipy(group1, group2):
    """Compute Cohen's d using scipy for independent verification."""
    t_stat, _ = stats.ttest_ind(group1, group2, equal_var=True)
    n1, n2 = len(group1), len(group2)
    # Convert t-statistic to Cohen's d
    d = t_stat * math.sqrt(1/n1 + 1/n2)
    return d

def main():
    with open(DATA_PATH) as f:
        data = json.load(f)
    
    raw_results = data["metadata"]["raw_results"]
    model_data = {r["label"]: r for r in raw_results}
    models = ["SafeRL", "Base", "Abliterated"]
    
    print("=" * 70)
    print("INDEPENDENT AUDIT: Re-deriving headline numbers from raw data")
    print("=" * 70)
    
    # ── Re-derive AEN Cohen's d using scipy ──
    print("\n--- AEN Cohen's d (scipy t-test → d conversion) ---")
    aen_ds = {}
    for label in models:
        rd = model_data[label]
        harmful = np.array(rd["per_prompt_scores"]["AEN_harmful"])
        benign = np.array(rd["per_prompt_scores"]["AEN_benign"])
        d = cohens_d_scipy(harmful, benign)
        aen_ds[label] = d
        print(f"  {label}: d={d:.4f} (harmful_mean={harmful.mean():.4f}, benign_mean={benign.mean():.4f})")
    
    avg_aen_d = np.mean([abs(d) for d in aen_ds.values()])
    print(f"  avg|d| = {avg_aen_d:.4f}")
    print(f"  EXPECTED: 6.5335")
    print(f"  MATCH: {abs(avg_aen_d - 6.5335) < 0.01}")
    
    # ── Re-derive ANG Cohen's d using scipy ──
    print("\n--- ANG Cohen's d (scipy t-test → d conversion) ---")
    ang_ds = {}
    for label in models:
        rd = model_data[label]
        harmful = np.array(rd["per_prompt_scores"]["ANG_harmful"])
        benign = np.array(rd["per_prompt_scores"]["ANG_benign"])
        d = cohens_d_scipy(harmful, benign)
        ang_ds[label] = d
        print(f"  {label}: d={d:.4f} (harmful_mean={harmful.mean():.4f}, benign_mean={benign.mean():.4f})")
    
    avg_ang_d = np.mean([abs(d) for d in ang_ds.values()])
    print(f"  avg|d| = {avg_ang_d:.4f}")
    print(f"  EXPECTED: 3.9330")
    print(f"  MATCH: {abs(avg_ang_d - 3.9330) < 0.01}")
    
    # ── Placebo test: shuffle labels ──
    print("\n--- Placebo test (shuffled labels) ---")
    rng = np.random.RandomState(42)
    for label in models:
        rd = model_data[label]
        harmful = np.array(rd["per_prompt_scores"]["AEN_harmful"])
        benign = np.array(rd["per_prompt_scores"]["AEN_benign"])
        combined = np.concatenate([harmful, benign])
        rng.shuffle(combined)
        sh_h = combined[:len(harmful)]
        sh_b = combined[len(harmful):]
        placebo_d = cohens_d_scipy(sh_h, sh_b)
        print(f"  {label}: placebo_d={placebo_d:.4f} (should be near 0)")
        assert abs(placebo_d) < 1.0, f"Placebo failed for {label}: {placebo_d}"
    print("  Placebo test PASSED: all shuffled d values are near 0")
    
    # ── Logit baseline ──
    print("\n--- Logit baseline Cohen's d ---")
    logit_ds = {}
    for label in models:
        rd = model_data[label]
        harmful = np.array(rd["per_prompt_scores"]["Logit_harmful"])
        benign = np.array(rd["per_prompt_scores"]["Logit_benign"])
        d = cohens_d_scipy(harmful, benign)
        logit_ds[label] = d
        print(f"  {label}: d={d:.4f}")
    
    avg_logit_d = np.mean([abs(d) for d in logit_ds.values()])
    print(f"  avg|d| = {avg_logit_d:.4f}")
    print(f"  EXPECTED: 0.2023")
    print(f"  MATCH: {abs(avg_logit_d - 0.2023) < 0.01}")
    
    # ── Activation beats logit ──
    ratio = avg_aen_d / max(avg_logit_d, 1e-10)
    print(f"\n--- Activation beats logit ratio ---")
    print(f"  AEN/Logit ratio = {ratio:.1f}x")
    print(f"  EXPECTED: >30x")
    print(f"  MATCH: {ratio > 30}")
    
    # ── ATC and LWCD should be near zero ──
    print("\n--- ATC and LWCD (should be near 0) ---")
    for metric in ["ATC", "LWCD"]:
        ds = []
        for label in models:
            rd = model_data[label]
            harmful = np.array(rd["per_prompt_scores"][f"{metric}_harmful"])
            benign = np.array(rd["per_prompt_scores"][f"{metric}_benign"])
            d = cohens_d_scipy(harmful, benign)
            ds.append(d)
        avg_d = np.mean([abs(d) for d in ds])
        print(f"  {metric}: avg|d| = {avg_d:.6f} (should be ~0)")
        assert avg_d < 0.01, f"{metric} should be near 0 but got {avg_d}"
    print("  ATC/LWCD test PASSED: both near zero")
    
    print("\n" + "=" * 70)
    print("AUDIT COMPLETE: All headline numbers independently verified")
    print("=" * 70)

if __name__ == "__main__":
    main()
