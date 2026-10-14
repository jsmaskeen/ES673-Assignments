from typing import Dict, List, Any, Optional


def load_wic(
    split: str = "validation",
    max_samples: Optional[int] = 200,
) -> List[Dict[str, Any]]:
    """
    Loads SuperGLUE Words in Context (WiC) polysemy dataset.
    Target word appears in sentence1 and sentence2; label indicates whether
    the word shares the same sense (1) or different senses (0).
    """
    from datasets import load_dataset
    try:
        ds = load_dataset("aps/super_glue", "wic", split=split)
    except Exception:
        ds = load_dataset("super_glue", "wic", split=split)
    items: List[Dict[str, Any]] = []

    for idx, row in enumerate(ds):
        if max_samples is not None and idx >= max_samples:
            break
        items.append({
            "id": f"wic_{idx}",
            "word": row["word"],
            "sentence1": row["sentence1"],
            "sentence2": row["sentence2"],
            "start1": int(row["start1"]),
            "end1": int(row["end1"]),
            "start2": int(row["start2"]),
            "end2": int(row["end2"]),
            "label": int(row["label"]),  # 1 = same sense, 0 = different sense
        })

    return items
