#!/usr/bin/env python3
"""Load raw safety datasets and convert to exp_sel_data_out.json schema.

Reads raw JSON files from temp/datasets/ and produces full_data_out.json
with each row as a separate example, grouped by dataset.
"""

from loguru import logger
from pathlib import Path
import json
import sys

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

WORKSPACE = Path(__file__).resolve().parent
TEMP_DIR = WORKSPACE / "temp" / "datasets"
OUTPUT = WORKSPACE / "full_data_out.json"


def load_json(path: Path) -> list[dict]:
    """Load a JSON array file."""
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array")
    return data


def convert_xstest(data: list[dict]) -> list[dict]:
    """Convert XSTest rows to schema format."""
    examples = []
    for i, item in enumerate(data):
        examples.append({
            "input": item["input"],
            "output": str(item["expected_refusal"]),
            "metadata_fold": 0 if item.get("metadata_fold") == "train" else 1,
            "metadata_category": item.get("category", "unknown"),
            "metadata_source": "Paul/XSTest",
            "metadata_source_id": item.get("source_id", i),
            "metadata_task_type": "classification",
            "metadata_n_classes": 2,
            "metadata_row_index": i,
        })
    return examples


def convert_air_bench(data: list[dict]) -> list[dict]:
    """Convert AIR-Bench rows to schema format."""
    examples = []
    for i, item in enumerate(data):
        # raw_air_bench.json already has 'input' key (pre-normalized)
        prompt = item.get("input", item.get("prompt", ""))
        expected = item.get("expected_refusal", True)
        category = item.get("category", "harmful")
        examples.append({
            "input": prompt,
            "output": str(expected),
            "metadata_fold": 0,
            "metadata_category": category,
            "metadata_source": "stanford-crfm/air-bench-2024",
            "metadata_source_id": item.get("source_id", i),
            "metadata_task_type": "classification",
            "metadata_n_classes": 2,
            "metadata_row_index": i,
        })
    return examples


def convert_jbb(data: list[dict]) -> list[dict]:
    """Convert JBB-Behaviors rows to schema format."""
    examples = []
    for i, item in enumerate(data):
        examples.append({
            "input": item["input"],
            "output": str(item["expected_refusal"]),
            "metadata_fold": 0,
            "metadata_category": item.get("category", "unknown"),
            "metadata_source": "JailbreakBench/JBB-Behaviors",
            "metadata_source_id": item.get("source_id", i),
            "metadata_task_type": "classification",
            "metadata_n_classes": 2,
            "metadata_row_index": i,
        })
    return examples


def convert_aegis(data: list[dict]) -> list[dict]:
    """Convert Aegis/Nemotron rows to schema format."""
    examples = []
    for i, item in enumerate(data):
        examples.append({
            "input": item["input"],
            "output": str(item["expected_refusal"]),
            "metadata_fold": 0,
            "metadata_category": item.get("category", "unknown"),
            "metadata_source": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
            "metadata_source_id": item.get("source_id", i),
            "metadata_task_type": "classification",
            "metadata_n_classes": 2,
            "metadata_row_index": i,
        })
    return examples


def convert_pku(data: list[dict]) -> list[dict]:
    """Convert PKU-SafeRLHF rows to schema format."""
    examples = []
    for i, item in enumerate(data):
        # Extract primary category from dict string
        cat_str = item.get("category", "")
        primary_cat = "unknown"
        if isinstance(cat_str, str) and cat_str.startswith("{"):
            try:
                cat_dict = eval(cat_str)
                true_cats = [k for k, v in cat_dict.items() if v]
                primary_cat = true_cats[0] if true_cats else "unknown"
            except Exception:
                primary_cat = "unknown"
        examples.append({
            "input": item["input"],
            "output": str(item["expected_refusal"]),
            "metadata_fold": 0,
            "metadata_category": primary_cat,
            "metadata_source": "PKU-SafeRLHF",
            "metadata_source_id": item.get("source_id", i),
            "metadata_task_type": "classification",
            "metadata_n_classes": 2,
            "metadata_row_index": i,
        })
    return examples


def convert_trustllm(data: list[dict]) -> list[dict]:
    """Convert TrustLLM rows to schema format."""
    examples = []
    for i, item in enumerate(data):
        examples.append({
            "input": item["input"],
            "output": str(item["expected_refusal"]),
            "metadata_fold": 0,
            "metadata_category": item.get("category", "jailbreak"),
            "metadata_source": "TrustLLM/TrustLLM-dataset",
            "metadata_source_id": item.get("source_id", i),
            "metadata_task_type": "classification",
            "metadata_n_classes": 2,
            "metadata_row_index": i,
        })
    return examples


@logger.catch(reraise=True)
def main():
    logger.info("Loading and converting top 3 safety datasets...")

    datasets = []

    # 1. XSTest (benign prompts, false refusal test) — TOP PICK
    xstest_path = TEMP_DIR / "raw_xstest.json"
    if xstest_path.exists():
        data = load_json(xstest_path)
        examples = convert_xstest(data)
        datasets.append({"dataset": "xstest", "examples": examples})
        logger.info(f"XSTest: {len(examples)} examples")
    else:
        logger.error("XSTest raw file not found at {xstest_path}")

    # 2. PKU-SafeRLHF (large mixed dataset) — TOP PICK
    pku_path = TEMP_DIR / "raw_pku_saferlhf.json"
    if pku_path.exists():
        data = load_json(pku_path)
        examples = convert_pku(data)
        datasets.append({"dataset": "pku_saferlhf", "examples": examples})
        logger.info(f"PKU-SafeRLHF: {len(examples)} examples")
    else:
        logger.error("PKU-SafeRLHF raw file not found at {pku_path}")

    # 3. JBB-Behaviors (harmful behaviors) — TOP PICK
    jbb_path = TEMP_DIR / "raw_jbb_behaviors.json"
    if jbb_path.exists():
        data = load_json(jbb_path)
        examples = convert_jbb(data)
        datasets.append({"dataset": "jbb_behaviors", "examples": examples})
        logger.info(f"JBB-Behaviors: {len(examples)} examples")
    else:
        logger.error("JBB-Behaviors raw file not found at {jbb_path}")

    # Build output
    output = {
        "metadata": {
            "description": "Safety prompt datasets for mechanistic interpretability analysis — top 3 datasets",
            "sources": ["Paul/XSTest", "PKU-SafeRLHF", "JailbreakBench/JBB-Behaviors"],
            "task": "safety_classification",
            "total_examples": sum(len(d["examples"]) for d in datasets),
            "selection_rationale": (
                "xstest: false refusal test with 18 linguistic categories; "
                "pku_saferlhf: 73K examples across 20 harm categories with balanced split; "
                "jbb_behaviors: clean 10-category harm taxonomy"
            ),
        },
        "datasets": datasets,
    }

    OUTPUT.write_text(json.dumps(output, indent=2))
    total = sum(len(d["examples"]) for d in datasets)
    logger.info(f"Saved {total} examples across {len(datasets)} datasets to {OUTPUT}")

    # Print summary
    for d in datasets:
        cats = {}
        for ex in d["examples"]:
            c = ex.get("metadata_category", "unknown")
            cats[c] = cats.get(c, 0) + 1
        logger.info(f"  {d['dataset']}: {len(d['examples'])} examples, {len(cats)} categories")


if __name__ == "__main__":
    main()
