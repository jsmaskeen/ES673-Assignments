from typing import Dict, List, Any, Optional


BLIMP_ADVANCED_SUBSETS: Dict[str, str] = {
    "coordinate_structure_constraint_object_extraction": "Island Constraints (Coordinate Structure)",
    "principle_A_domain_1": "Binding Theory (Principle A Domain)",
    "principle_A_c_command": "Binding Theory (Principle A C-Command)",
    "wh_questions_object_gap": "Filler-Gap Dependencies (Object Gap)",
    "wh_island": "Island Constraints (WH-Islands)",
    "adjunct_island": "Island Constraints (Adjuncts)",
    "passive_1": "Argument Alternation (Passives)",
    "animate_subject_trans": "Selectional Restrictions (Animacy)",
    "npi_present_1": "Negative Polarity Licensing (NPI)",
    "irregular_past_participle_verbs": "Morphological Irregularity",
}


def load_blimp_subset(
    subset_name: str,
    split: str = "train",
    max_samples: Optional[int] = 100,
) -> List[Dict[str, Any]]:
    """
    Loads BLiMP minimal pair data for a target linguistic phenomenon.
    Each item contains sentence_good, sentence_bad, and phenomenon metadata.
    """
    if subset_name not in BLIMP_ADVANCED_SUBSETS:
        raise ValueError(f"Unknown BLiMP subset: {subset_name}. Supported: {list(BLIMP_ADVANCED_SUBSETS.keys())}")

    from datasets import load_dataset
    ds = load_dataset("nyu-mll/blimp", subset_name, split=split)
    items: List[Dict[str, Any]] = []

    for idx, row in enumerate(ds):
        if max_samples is not None and idx >= max_samples:
            break
        items.append({
            "id": f"{subset_name}_{idx}",
            "subset": subset_name,
            "category": BLIMP_ADVANCED_SUBSETS[subset_name],
            "sentence_good": row["sentence_good"],
            "sentence_bad": row["sentence_bad"],
            "field": row.get("field", BLIMP_ADVANCED_SUBSETS[subset_name]),
            "linguistics_term": row.get("linguistics_term", subset_name),
        })

    return items
