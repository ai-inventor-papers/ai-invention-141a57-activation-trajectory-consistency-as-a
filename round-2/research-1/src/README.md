# Safety Benchmark Research & Literature Review

Research artifact gathering ground-truth safety benchmark scores and reviewing prior activation-based safety metrics for the single-model safety metric project.

## What This Does

Conducts comprehensive web research to collect:
1. **Ground-truth safety benchmark scores** for 10+ models across Qwen, Llama, Mistral, and Gemma families from official sources (TrustLLM, AIR-Bench, XSTest, model cards)
2. **Literature review** of all prior activation-based safety methods (SIREN, AMS, latent-space probes) to establish novelty gaps
3. **Model lineage documentation** for Qwen3-4B family (Base → Instruct → SafeRL → Abliterated)

## Layout

| File | Description |
|------|-------------|
| `.sdk_openhands_agent_struct_out.json` | Main research output: 25 cited sources, benchmark tables, novelty gap analysis |
| `README.md` | This file |
| `.aii/manifest.yaml` | Storage manifest (no heavy files in this artifact) |

## Key Findings

### Benchmark Scores Collected
- **Qwen3-4B-SafeRL**: 86.5% safety (Qwen3-235B judge), 98.1% (WildGuard), 5.3% over-refusal
- **Qwen3-4B (baseline)**: 47.5% safety, 64.7% (WildGuard), 12.9% over-refusal
- **XSTest**: Over-refusal ranges from 0.8% (Mistral-Instruct) to 38% (Llama2-70B-Chat)
- **AIR-Bench**: Claude 3 Sonnet 89% refusal, DBRX Instruct 15% refusal

### Prior Activation-Based Methods
| Method | Requires | Prompts | Training | Novelty Gap |
|--------|----------|---------|----------|-------------|
| SIREN (Jiao et al. 2026) | Labeled data + classifier head | Full dataset | Yes (linear probes) | Needs labels, training |
| AMS (Google 2026) | Contrastive prompt pairs | Many pairs | No | Needs contrastive pairs |
| Latent Probes (Khatri et al. 2026) | Full dataset + MLP training | Full dataset | Yes (MLP) | Needs labels, training |
| **Proposed (AEN/ANG)** | **Nothing** | **3-5 prompts** | **No** | **Zero-shot, reference-free** |

## How to Use

This artifact produces research output consumed by downstream experiment artifacts. No code to run — the JSON output is read directly by the next pipeline step.

```bash
# Read the research output
cat .sdk_openhands_agent_struct_out.json | python -m json.tool | head -50
```

## Sources

25 sources consulted including:
- Official model cards (Qwen3-4B, Qwen3-4B-SafeRL, abliterated variants)
- Peer-reviewed papers (SIREN ACL 2026, XSTest NAACL 2024, AIR-Bench 2024)
- Technical reports (Qwen3 Technical Report, Qwen3Guard)
- Blog posts (Google AMS, HuggingFace abliteration guide)
- Leaderboards (TrustLLM, XSTest, AIR-Bench)

## Restoring Removed Files

The `hf_cache/` directory was removed after this run. It is redownloadable:

```bash
# Restore HuggingFace cache for any specific model needed
huggingface-cli download <model-id> --cache-dir hf_cache/
```

No models were actually cached during this research-only artifact, so the directory is empty and restoration is not needed in practice.
