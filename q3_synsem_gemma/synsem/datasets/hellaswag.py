from typing import Dict, List, Any, Optional


def load_hellaswag(
    split: str = "validation",
    max_samples: Optional[int] = 100,
) -> List[Dict[str, Any]]:
    """
    Loads HellaSwag commonsense physical reasoning dataset.
    Given context, selects the most plausible physical continuation out of 4 endings.
    """
    from datasets import load_dataset
    try:
        ds = load_dataset("Rowan/hellaswag", split=split)
    except Exception:
        ds = load_dataset("hellaswag", split=split)
    items: List[Dict[str, Any]] = []

    for idx, row in enumerate(ds):
        if max_samples is not None and idx >= max_samples:
            break
        items.append({
            "id": f"hellaswag_{idx}",
            "activity_label": row.get("activity_label", "general"),
            "ctx": row["ctx"],
            "endings": row["endings"],
            "label": int(row["label"]),  # 0, 1, 2, or 3
        })

    return items
