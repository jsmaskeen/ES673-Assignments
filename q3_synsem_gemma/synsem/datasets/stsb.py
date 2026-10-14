from typing import Dict, List, Any, Optional


def load_stsb(
    split: str = "validation",
    max_samples: Optional[int] = 200,
) -> List[Dict[str, Any]]:
    """
    Loads STS-B semantic textual similarity sentence pairs with continuous
    gold similarity ratings (0.0 = completely unrelated, 5.0 = semantically identical).
    """
    from datasets import load_dataset
    try:
        ds = load_dataset("nyu-mll/glue", "stsb", split=split)
    except Exception:
        ds = load_dataset("glue", "stsb", split=split)
    items: List[Dict[str, Any]] = []

    for idx, row in enumerate(ds):
        if max_samples is not None and idx >= max_samples:
            break
        items.append({
            "id": f"stsb_{idx}",
            "sentence1": row["sentence1"],
            "sentence2": row["sentence2"],
            "score": float(row["label"]),  # 0.0 to 5.0
            "normalized_score": float(row["label"]) / 5.0,  # 0.0 to 1.0
        })

    return items
