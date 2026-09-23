# Safety Prompt Dataset for Mechanistic Interpretability

Curated dataset of harmful and benign prompts from 6 published safety benchmarks, designed for mechanistic interpretability analysis of model activations on safety-related inputs.

## Layout

| File | Description |
|---|---|
| `full_data_out.json` | Full dataset: 127,504 examples across 6 datasets (exp_sel_data_out schema) |
| `mini_full_data_out.json` | First 3 examples per dataset (18 total) |
| `preview_full_data_out.json` | First 3 examples per dataset with truncated strings |
| `data_out.json` | Curated subset: 118 prompts (100 train + 18 heldout) |
| `mini_data_out.json` | First 3 curated prompts |
| `preview_data_out.json` | First 3 curated prompts with truncated strings |
| `data.py` | Script to convert raw downloads to exp_sel_data_out schema |
| `curate_dataset.py` | Script to produce focused curated subset |
| `temp/datasets/raw_*.json` | Raw downloads from each source dataset |

## Schema

Each entry is a JSON object with:
- `input`: The prompt text (max 512 tokens)
- `output`: `"True"` for harmful (expected refusal), `"False"` for benign
- `metadata_fold`: `0` = train, `1` = heldout
- `metadata_category`: Harm category or prompt type
- `metadata_source`: HuggingFace dataset ID
- `metadata_task_type`: `"classification"`
- `metadata_n_classes`: `2`

## Dataset Composition (full_data_out.json)

| Dataset | Examples | Categories | Role |
|---|---|---|---|
| **xstest** | 450 | 18 | False refusal test (benign prompts that look dangerous) |
| **pku_saferlhf** | 73,907 | 20 | Largest mixed dataset with diverse harm types |
| **aegis** | 29,095 | 2 | NVIDIA-labeled safe/unsafe prompts |
| **trustllm** | 18,258 | 1 | Jailbreak prompts |
| **air_bench** | 5,694 | 1 | Malicious prompts (Stanford CRFM) |
| **jbb_behaviors** | 100 | 10 | Well-defined harm taxonomy |

## Top 3 Datasets for Mech Interp

1. **xstest** — Essential for false refusal testing; 18 linguistic categories (homonyms, figurative language, etc.)
2. **pku_saferlhf** — 73K examples across 20 harm categories with balanced harmful/benign split
3. **jbb_behaviors** — Clean 10-category taxonomy (Harassment, Malware, Physical harm, Economic harm, etc.)

## Source Datasets

| Source | Downloads | Provenance |
|---|---|---|
| Paul/XSTest | 4,384 | NAACL 2024 |
| stanford-crfm/air-bench-2024 | 7,708 | Stanford CRFM |
| JailbreakBench/JBB-Behaviors | 50,896 | JailbreakBench paper |
| nvidia/Aegis-AI-Content-Safety-Dataset-2.0 | 8,324 | NVIDIA Nemotron |
| PKU-SafeRLHF | 14,501 | PKU paper |
| TrustLLM/TrustLLM-dataset | 133 | TrustLLM paper |

## How to Run

```bash
# Convert raw downloads to schema format
python data.py

# Produce curated subset
python curate_dataset.py
```

## Restoring Removed Files

The `temp/datasets/` and `logs/` directories are marked for deletion. Restore with:

```bash
# Install dependencies
uv pip install datasets loguru

# Download raw datasets
export SKILL_DIR="/ai-inventor/.claude/skills/aii-hf-datasets"
export PY="$SKILL_DIR/../.ability_client_venv/bin/python"
export S="$SKILL_DIR/scripts/aii_hf_download_datasets.py"
mkdir -p temp/datasets

$PY $S stanford-crfm/air-bench-2024 --split test --output-dir temp/datasets
$PY $S nvidia/Aegis-AI-Content-Safety-Dataset-2.0 --split test --output-dir temp/datasets
$PY $S declare-lab/HarmfulQA --split train --output-dir temp/datasets
$PY $S mlabonne/harmful_behaviors --split train --output-dir temp/datasets
$PY $S walledai/TDC23-RedTeaming --split train --output-dir temp/datasets
$PY $S allenai/tulu-3-trustllm-jailbreaktrigger-eval --split test --output-dir temp/datasets
$PY $S Paul/XSTest --split train --output-dir temp/datasets

# Then re-run
python data.py
```

## Notes

- All prompts are English, <512 tokens
- Binary classification: expected_refusal=True (harmful) vs False (benign)
- XSTest benign prompts are critical for testing false refusal
- JBB-Behaviors provides clean harm taxonomy
- PKU-SafeRLHF provides scale and category diversity
