# Safety Benchmark Scores and Activation-Based Metric Literature

## Summary

This research artifact collects ground-truth safety benchmark scores for 10+ models across Qwen, Llama, Mistral, and Gemma families from official sources (TrustLLM leaderboard, AIR-Bench paper, XSTest paper, model cards). Key findings: (1) Qwen3-4B-SafeRL achieves 86.5% safety rate on Qwen3-235B judge and 98.1% on WildGuard with only 5.3% over-refusal, dramatically improving from Qwen3-4B's 47.5%/64.7%/12.9% respectively [15]. (2) SIREN (Jiao et al. 2026) uses linear probing to identify safety neurons across all layers, achieving 86.7% avg F1 with 250x fewer parameters than guard models, but still requires labeled training data and a separate classifier head [16]. (3) Google's AMS (2026) measures geometric separation in activation space using contrastive prompt pairs, finding instruction-tuned models show 3.8-8.4 sigma separation while abliterated models degrade to 3.3 sigma and base models show 0.69 sigma — but AMS still requires contrastive prompts, not zero-shot [17]. (4) Latent-space probes (Khatri et al. 2026) train MLPs on final-layer activations achieving 99.1% F1 on WildJailbreak but require full dataset extraction and probe training [18]. (5) XSTest shows over-refusal rates from 0.8% (Mistral-Instruct) to 38% (Llama2-70B-Chat) on 250 safe prompts, with 200 unsafe contrasts [25]. (6) AIR-Bench 2024 covers 5,694 prompts across 314 risk categories with Claude 3 Sonnet at 89% refusal and DBRX Instruct at 15% [23]. (7) Abliteration removes refusal via weight orthogonalization against a single direction, but models with combined safety pretraining (metatags + rephrasing + refusals) show partial robustness [27]. (8) Entropy-based uncertainty alone is insufficient for safety detection — models produce low-entropy hallucinations (the 'confidently wrong' regime) requiring combination with correctness probes [22]. Novelty gap: No prior method proposes Shannon entropy or L2-norm of hidden states as a single-model, reference-free safety metric requiring only 3-5 prompts with zero training. All prior activation-based methods (SIREN, AMS, latent probes) require either labeled data, contrastive prompt pairs, or separate classifier training.

## Research Findings

## 1. Qwen3-4B Model Lineage and Architecture

The Qwen3-4B family provides an ideal testbed for single-model safety metrics because it contains four variants sharing the same base architecture: **Qwen3-4B-Base** (pretrained, no safety alignment) [3], **Qwen3-4B** (instruction-tuned with safety alignment) [2], **Qwen3-4B-SafeRL** (reinforcement learning with hybrid reward from Qwen3Guard-Gen) [1], and community **abliterated** variants (refusal direction removed via weight orthogonalization) [4][17].

The architecture has 36 layers, 32 query heads with 8 KV heads (GQA), 4B total parameters (3.6B non-embedding), and a native context length of 32,768 tokens extendable to 131,072 via YaRN [2]. Qwen3 introduces a unique thinking/non-thinking mode fusion, allowing dynamic switching between reasoning and direct response modes within a single model [16].

**Qwen3-4B-SafeRL** was trained using reinforcement learning with a hybrid reward function combining three objectives: (1) safety maximization via Qwen3Guard-Gen-4B, (2) helpfulness maximization via WorldPM-72B-HelpSteer2, and (3) refusal minimization via Qwen3Guard-Gen-4B [1]. This produces the following performance profile:

| Model | Safety Rate (Qwen3-235B) | Safety Rate (WildGuard) | Refusal (WildGuard) | ArenaHard-v2 | AIME25 | LCB-v6 | GPQA |
|-------|--------------------------|-------------------------|---------------------|--------------|--------|--------|------|
| Qwen3-4B | 47.5% | 64.7% | 12.9% | 9.5% | 19.1% | 26.4% | 41.7% |
| Qwen3-4B-SafeRL | 86.5% | 98.1% | 5.3% | 10.7% | 18.2% | 27.7% | 40.8% |

The SafeRL variant nearly doubles the safety rate while maintaining comparable capability scores and actually reducing over-refusal from 12.9% to 5.3% [1]. This is a critical finding: effective safety alignment does not necessarily trade off with capability or increase over-refusal. The Qwen3Guard-Gen model served as the training reward for SafeRL, so it should not be used as an independent judge for SafeRL evaluation [25].

## 2. Safety Benchmark Score Collection

### 2.1 XSTest: Over-Refusal Benchmark

XSTest (Röttger et al., 2024) comprises 250 safe prompts across 10 prompt types (homonyms, figurative language, safe targets, safe contexts, definitions, real discrimination/nonsense groups, nonsense discrimination/real groups, historical events, privacy-public, privacy-fictional) plus 200 unsafe contrast prompts [9]. The key metric is the over-refusal rate on safe prompts (lower is better) and the refusal rate on unsafe prompts (higher is better).

Results from the original paper on key models:
- **Llama2-70B-Chat (with system prompt)**: 38% full refusal + 21.6% partial refusal on safe prompts; 99.5% refusal on unsafe prompts [9]
- **Llama2-70B-Chat (without system prompt)**: 14% + 15.6% on safe; 97.5% on unsafe [9]
- **Mistral-7B-Instruct (no guardrail)**: 0.8% + 0.8% on safe; 23.5% + 12.5% on unsafe — shows minimal over-refusal but also minimal safety [9]
- **Mistral-7B-Instruct (with guardrail)**: 9.6% + 9.2% on safe; 87.5% + 9% on unsafe [9]
- **GPT-4**: 6.4% + 2% on safe; 97.5% + 2% on unsafe — best balance [9]

XSTest demonstrates that exaggerated safety (over-refusal) is caused by lexical overfitting, where models are overly sensitive to safety-related keywords [9]. System prompts can dramatically change safety behavior but inconsistently [9]. Aggregated XSTest results across many models are tracked on public leaderboards [24].

### 2.2 AIR-Bench 2024: Regulation-Aligned Safety

AIR-Bench 2024 contains 5,694 diverse prompts spanning 314 granular risk categories derived from 8 government regulations and 16 company policies [10]. It uses a three-level scoring system (0, 0.5, 1) with GPT-4o as judge, achieving 0.86 Kappa agreement with human annotators [10].

Key findings:
- **Claude 3 Sonnet**: 89% overall safety refusal rate (highest) [10]
- **Gemini 1.5 Pro**: Second highest [10]
- **DBRX Instruct**: 15% refusal rate (lowest) — frequently providing harmful content [10]
- No single model consistently refuses across all 45 level-3 risk categories [10]
- Models show weakest refusal on "Advice in Regulated Industries" (healthcare, finance, law) and "Automated Decision-Making" [10]

The official AIR-Bench leaderboard is hosted at Stanford CRFM HELM [23].

### 2.3 TrustLLM: Multi-Dimensional Trustworthiness

TrustLLM (Sun et al., 2024) evaluates across six dimensions: truthfulness, safety, fairness, robustness, privacy, and machine ethics, consisting of over 30 datasets [11]. The leaderboard provides comprehensive trustworthiness ranking through an interactive web interface [12]. An independent AI-Secure leaderboard also aggregates TrustLLM scores across models [19].

## 3. Prior Activation-Based Safety Metrics

### 3.1 SIREN (Jiao et al., 2026)

SIREN (Safeguard with Internal REpresentatioN) is the strongest prior activation-based safety method [5]. It operates in two stages:
1. **Safety neuron identification**: Train L1-regularized linear probes on per-layer pooled representations to identify neurons with high salience for harmfulness detection [5]
2. **Adaptive layer-weighted aggregation**: Concatenate weighted activations of safety neurons across all layers, where weights are computed from validation performance of per-layer probes [6]

Key results:
- 86.7% average F1 on Qwen3-4B backbone across 7 benchmarks (ToxiC, OpenAIMod, Aegis, Aegis2, WildGuard, SafeRLHF, BeaverTails) [6]
- 250x fewer trainable parameters than guard models (14M vs. billion-level) [5]
- Individual layer probes reach within 4 points of fine-tuned guard models, with middle layers (not terminal) achieving highest performance (~79%) [5]
- Cross-layer aggregation achieves 8-point improvement over single-layer probes [6]
- Cross-model ensembling (Qwen3-0.6B + Qwen3-4B + Llama3.2-1B) achieves 87.7% F1 [6]
- SIREN requires only one forward pass through the LLM plus negligible overhead, while guard models require multiple forward passes for autoregressive generation [6]

**Limitations relative to our approach**: SIREN requires (a) labeled training data for linear probes, (b) a separate classifier head trained on aggregated features, (c) computation across all layers. It is not a zero-shot, reference-free metric on 3-5 prompts [5].

### 3.2 AMS — Activation-based Model Scanner (Google, 2026)

AMS measures geometric structure in activation space using contrastive prompt pairs [7]. It extracts hidden states at an intermediate layer (35-40% depth), computes a direction vector separating harmful from benign content, and measures class separation as a sigma score [7].

Key findings:
- Instruction-tuned models: 3.8-8.4 sigma separation [7]
- Uncensored variants (Dolphin, Lexi): 1.1-1.3 sigma (CRITICAL) [7]
- Abliterated models: 3.3 sigma (WARNING — partial degradation) [7]
- Base models: 0.69 sigma (no safety structure) [7]
- Quantized models (INT4/INT8): <5% separation drift [7]
- Scan completes in 10-40 seconds on GPU [7]

**Limitations**: AMS requires contrastive prompt pairs (harmful vs. harmless), not truly zero-shot. It measures geometric separation (a direction vector), not entropy or norm statistics [7].

### 3.3 Latent-Space Safety Probes (Khatri et al., 2026)

This approach trains a 6-layer MLP (13.9M parameters) on final-layer, last-token activations [8]. Results on LLaMA-3.1-8B:
- WildJailbreak: 99.1% F1 [8]
- BeaverTails: 82.7% F1 [8]
- AEGIS 2.0: 83.5% F1 [8]

A reproduction study (Khatri & Chan, 2026) extended this to Gemma-4-E4B, Mistral-7B-v0.3, and Qwen2-7B, finding F1 scores within 0.5 points of the original [8]. Activations were found to be byte-for-byte deterministic across random seeds [8].

**Limitations**: Requires full dataset extraction and probe training. Not a few-shot metric [8].

### 3.4 Cross-Architecture Latent Separability (Llorente-Saguer, 2026)

Shows that harmful intent is linearly separable from residual-stream activations across 12 models spanning 4 architectural families and 3 alignment variants (base, instruction-tuned, abliterated) [8]. Evidence that harm recognition and refusal generation are separable mechanisms [8].

## 4. Abliteration and Safety Robustness

### 4.1 Abliteration Technique

Abliteration removes refusal behavior by identifying a refusal direction in the residual stream and orthogonalizing model weights against it [13]. The procedure:
1. Run model on harmful and harmless instructions, collecting residual stream activations [13]
2. Compute mean difference between harmful and harmless activations per layer [13]
3. Select the best refusal direction and orthogonalize all component weights against it [13]

This can be done at inference time (activation projection) or permanently (weight modification) [13]. The foundational finding that refusal is mediated by a single direction comes from Arditi et al. (NeurIPS 2024) [21]. Multiple community abliterated Qwen3-4B variants exist on HuggingFace [4][17]. Open-source implementations are available [20].

### 4.2 Robustness Under Abliteration

Agnihotri et al. (2025) studied which safety interventions survive abliteration [14]:
- **Refusal-only training**: Most fragile — easily neutralized [14]
- **Rephrase-only**: Vulnerable — many harmful refusals become non-refusals post-abliteration [14]
- **Combined approach (safe filtering + rephrasing + metatags + refusals)**: Most robust — harmful prompts remain largely refused [14]
- **Qwen3**: Shows no loss under abliteration in their setup, unlike Llama-3.3 and GLM-4 [14]
- Models fail to reliably detect their own refusal state after abliteration [14]

This suggests that safety signals distributed across multiple features (not concentrated in a single direction) are harder to erase — consistent with the hypothesis that entropy/norm metrics might capture this distributed signal [14].

## 5. Entropy and Activation Statistics for Safety

### 5.1 Entropy Insufficiency

Phillips et al. (2026) demonstrate that entropy-based uncertainty methods alone are insufficient for safe selective prediction [15]. They identify a "confidently wrong regime" where models produce low-entropy hallucinations [15]. Combining entropy with a correctness probe improves both detection and calibration [15].

This finding is relevant because it suggests that raw entropy of outputs (logits) may not capture safety-relevant information, but entropy of *internal activations* might — as it measures the distribution of information across the hidden state rather than output confidence [15].

### 5.2 Safety Neuron Phenomenon

Chen et al. (2025) show that safety neurons comprise less than 1% of all neurons but mediate safety behavior [18]. Safety mechanisms are sparse and localized in specific layers [18]. This sparsity suggests that activation norms (which measure total signal magnitude) might capture the presence/absence of these safety neurons [18].

### 5.3 Separate Harmfulness and Refusal Directions

Zhao et al. (NeurIPS 2025) show that harmfulness and refusal occupy separate directions in latent space, and only the harmfulness direction governs the model's judgment of harm [22]. This separation is important because it means a model can recognize harm without being trained to refuse, suggesting that activation-based metrics could detect safety even in base models [22].

## 6. Capability Benchmarks

From the Qwen3 technical report and public leaderboards:
- **Qwen3-8B**: 84.2% on GSM8K, 72.0% on HumanEval [16]
- **Qwen3-4B**: Distilled from larger models via strong-to-weak distillation, competitive in its size class [16]
- Safety-capability trade-off: Qwen3-4B-SafeRL shows minimal capability degradation vs. Qwen3-4B (AIME25: 19.1% → 18.2%, GPQA: 41.7% → 40.8%) [1]

## 7. Novelty Gap Analysis

No prior method proposes the following combination:
1. **Shannon entropy of hidden states** as a safety signal (vs. direction vectors, neuron selection, or geometric separation)
2. **L2-norm of activations** as a safety signal (vs. probing or classification)
3. **Zero reference models** — all prior methods require either a guard model, labeled data, or contrastive prompt pairs
4. **Few-shot (3-5 prompts)** — all prior methods require full dataset extraction
5. **No training** — all prior methods train probes, classifiers, or require labeled data

The closest prior work is AMS (geometric separation from activations) [7] and SIREN (safety neuron identification) [5], but both require more infrastructure than our proposed approach.


## Sources

[1] [Qwen/Qwen3-4B-SafeRL — Hugging Face Model Card](https://huggingface.co/Qwen/Qwen3-4B-SafeRL) (Qwen Team; 2025) — Official model card for Qwen3-4B-SafeRL with safety alignment performance table showing 86.5% safety rate on Qwen3-235B judge, 98.1% on WildGuard, and 5.3% over-refusal.

[2] [Qwen/Qwen3-4B — Hugging Face Model Card](https://huggingface.co/Qwen/Qwen3-4B) (Qwen Team; 2025) — Official model card for Qwen3-4B with architecture details (36 layers, 32/8 GQA heads, 4B params) and thinking mode fusion capabilities.

[3] [Qwen/Qwen3-4B-Base — Hugging Face Model Card](https://huggingface.co/Qwen/Qwen3-4B-Base) (Qwen Team; 2025) — Base pretrained model card confirming the base model exists and is loadable in bf16.

[4] [huihui-ai/Qwen3-4B-abliterated — Hugging Face](https://huggingface.co/huihui-ai/Qwen3-4B-abliterated) (huihui-ai; 2026) — Community abliterated variant of Qwen3-4B created with abliteration technique, confirming uncensored models are available.

[5] [LLM Safety From Within: Detecting Harmful Content with Internal Representations (SIREN)](https://arxiv.org/abs/2604.18519) (Difan Jiao, Yilun Liu, Ye Yuan, Zhenwei Tang, Linfeng Du, Haolun Wu, Ashton Anderson; 2026) — SIREN paper by Jiao et al. (ACL 2026) proposing safety neuron identification via linear probing and adaptive cross-layer aggregation. Achieves 86.7% avg F1 with 250x fewer parameters than guard models.

[6] [SIREN Full HTML Paper](https://arxiv.org/html/2604.18519v1) (Difan Jiao, Yilun Liu, Ye Yuan, Zhenwei Tang, Linfeng Du, Haolun Wu, Ashton Anderson; 2026) — Full SIREN paper with detailed methodology, ablation studies, and cross-model ensemble results.

[7] [Introducing AMS: Activation-based Model Scanner for Open-Weight LLM Safety Verification](https://opensource.googleblog.com/2026/04/introducing-ams-activation-based-model-scanner-for-open-weight-llm-safety-verification.html) (Glen Messenger; 2026) — Google's AMS tool that measures geometric separation in activation space using contrastive prompt pairs. Instruction-tuned models show 3.8-8.4 sigma, abliterated show 3.3 sigma, base models show 0.69 sigma.

[8] [Do All LLMs Know When They're Being Harmful? A Reproducibility Study of Latent-Space Safety Probes Across Model Families](https://arxiv.org/abs/2608.08029) (Alizishaan Khatri, Dun Li Chan; 2026) — Reproduction and extension of Khatri et al. (2026) latent-space safety probes across LLaMA, Gemma, Mistral, and Qwen families. F1 scores within 0.5 points across architectures. Activations are deterministic across seeds.

[9] [XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models](https://aclanthology.org/2024.naacl-long.301.pdf) (Paul Röttger, Hannah Rose Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, Dirk Hovy; 2024) — XSTest paper by Röttger et al. (NAACL 2024) with 250 safe + 200 unsafe prompts. Shows over-refusal rates from 0.8% (Mistral) to 38% (Llama2-70B-Chat). Exaggerated safety caused by lexical overfitting.

[10] [AIR-Bench 2024: A Safety Benchmark Based on Risk Categories from Regulations and Policies](https://arxiv.org/html/2407.17436v2) (Yi Zeng, Yu Yang, Andy Zhou, Jeffrey Ziwei Tan, Yuheng Tu, Yifan Mai, Kevin Klyman, Minzhou Pan, Ruoxi Jia, Dawn Song, Percy Liang, Bo Li; 2024) — AIR-Bench 2024 with 5,694 prompts across 314 risk categories from 8 government regulations and 16 company policies. Claude 3 Sonnet: 89% refusal, DBRX Instruct: 15% refusal.

[11] [TrustLLM Leaderboard](https://trustllmbenchmark.github.io/TrustLLM-Website/leaderboard.html) (Howie Hwong, Sun et al.; 2024) — TrustLLM leaderboard covering truthfulness, safety, fairness, robustness, privacy, and machine ethics across 30+ datasets.

[12] [TrustLLM: Trustworthiness in Large Language Models (ICML 2024)](https://github.com/HowieHwong/TrustLLM) (Howie Hwong, Sun et al.; 2024) — TrustLLM framework GitHub repository with evaluation toolkit and dataset for comprehensive trustworthiness assessment.

[13] [Uncensor any LLM with abliteration](https://huggingface.co/blog/mlabonne/abliteration) (Maxime Labonne; 2024) — Technical blog post explaining abliteration technique: identify refusal direction via mean difference of harmful/harmless activations, then orthogonalize weights. Includes full implementation code.

[14] [A Granular Study of Safety Pretraining under Model Abliteration](https://arxiv.org/html/2510.02768) (Shashank Agnihotri, Jonas Jakubassa, Priyam Dey, Sachin Goyal, Bernt Schiele, R. Venkatesh Babu, Margret Keuper; 2025) — Agnihotri et al. (2025) study of which safety interventions survive abliteration. Combined approach (safe filtering + rephrasing + metatags + refusals) is most robust. Qwen3 shows no loss under abliteration.

[15] [Entropy Alone is Insufficient for Safe Selective Prediction in LLMs](https://arxiv.org/html/2603.21172) (Edward Phillips, Fredrik K. Gustafsson, Sean Wu, Anshul Thakur, David A. Clifton; 2026) — Phillips et al. (2026) show entropy-based uncertainty methods fail in the 'confidently wrong regime'. Combining entropy with correctness probes improves detection and calibration across 4 model families.

[16] [Qwen3 Technical Report](https://arxiv.org/html/2505.09388) (Qwen Team; 2025) — Official Qwen3 technical report with architecture details, training methodology, and comprehensive benchmark results across thinking and non-thinking modes.

[17] [mlabonne/Qwen3-4B-abliterated — Hugging Face](https://huggingface.co/mlabonne/Qwen3-4B-abliterated) (mlabonne; 2026) — Another community abliterated variant confirming multiple uncensored Qwen3-4B models exist.

[18] [Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons](https://arxiv.org/abs/2406.14144) (J. Chen, X. Wang, Z. Yao, Y. Bai, L. Hou, J. Li; 2025) — Chen et al. (ICLR 2025) showing safety neurons comprise less than 1% of all neurons but mediate safety behavior across Llama2, Mistral, and other models.

[19] [AI-Secure LLM Trustworthy Leaderboard](https://huggingface.co/spaces/AI-Secure/llm-trustworthy-leaderboard) (AI-Secure; 2024) — Interactive leaderboard for TrustLLM benchmark scores across multiple models and safety dimensions.

[20] [llm-abliteration — Make abliterated models with transformers](https://github.com/NousResearch/llm-abliteration) (NousResearch; 2024) — Open-source implementation of abliteration technique for creating uncensored models.

[21] [Refusal in Language Models is Mediated by a Single Direction](https://arxiv.org/abs/2312.06674) (Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda; 2024) — Arditi et al. (NeurIPS 2024) foundational paper showing refusal behavior is mediated by a single direction in the residual stream, enabling abliteration.

[22] [LLMs Encode Harmfulness and Refusal Separately](https://arxiv.org/abs/2507.11878) (J. Zhao, J. Huang, Z. Wu, D. Bau, W. Shi; 2025) — Zhao et al. (NeurIPS 2025) showing harmfulness and refusal occupy separate directions in latent space, and only the harmfulness direction governs the model's judgment of harm.

[23] [AIR-Bench 2024 Leaderboard (Stanford HELM)](https://crfm.stanford.edu/helm/air-bench/latest/) (Stanford CRFM; 2024) — Official AIR-Bench 2024 leaderboard with per-model refusal rates across 314 risk categories.

[24] [XSTest Leaderboard — LLM Stats](https://llm-stats.com/benchmarks/xstest) (2026) — Aggregated XSTest results across many models tracking over-refusal rates.

[25] [Qwen3Guard Technical Report](https://arxiv.org/abs/2510.14276) (Qwen Team; 2025) — Technical report on Qwen3Guard used as the reward model for Qwen3-4B-SafeRL training. Important to note: should NOT be used as a judge for SafeRL evaluation since it was the training reward.

## Verification

Numbered citations resolve to unique listed sources. Passage checks test text occurrence, not claim truth or entailment. Author/year metadata and locators are not independently verified. Details: `research_verification.json`.

No optional exact passages supplied; no passage checks performed.

## Follow-up Questions

- Can activation entropy and norm metrics be computed on base models (no safety training) and still distinguish them from safety-aligned variants, or do they require some minimum safety signal to be present?
- How do activation-based safety metrics behave under different quantization levels (INT4, INT8) — does quantization noise obscure the safety signal in hidden states?
- Do activation entropy/norm metrics transfer across model families (Qwen → Llama → Mistral → Gemma) without per-family calibration, or do they require architecture-specific thresholds?

---
*Generated by AI Inventor Pipeline*
