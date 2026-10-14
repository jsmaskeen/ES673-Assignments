import random
from typing import Dict, List, Any


SYNTACTIC_TEMPLATES: List[str] = [
    "The {adj} {subj} {verb} the {obj} {adv}.",
    "A {adj} {subj} {verb} {adv} in the room.",
    "The {subj} {verb} a {adj} {obj}.",
    "Every {adj} {subj} {verb} near the {obj}.",
    "That {subj} often {verb} {adv}.",
]

COHERENT_VOCAB = {
    "adj":  ["curious", "small", "ancient", "brave", "gentle", "diligent", "hungry", "clever"],
    "subj": ["scientist", "child", "detective", "gardener", "student", "teacher", "doctor", "artist"],
    "verb": ["examines", "discovers", "creates", "inspects", "protects", "observes", "studies", "measures"],
    "obj":  ["evidence", "solution", "painting", "specimen", "artifact", "telescope", "formula", "manuscript"],
    "adv":  ["carefully", "silently", "eagerly", "patiently", "precisely", "quickly", "steadily", "calmly"],
}

INCOHERENT_VOCAB = {
    "adj":  ["colorless", "Tuesday", "prime", "rectangular", "quadratic", "logarithmic", "cubic", "fractional"],
    "subj": ["ideas", "velocity", "grammar", "algebra", "silence", "Thursday", "symmetry", "distance"],
    "verb": ["sleep", "evaporates", "multiplies", "orbits", "crystallizes", "fractures", "digests", "dissolves"],
    "obj":  ["a velocity", "the Thursday", "some grammar", "triangles", "parabolas", "infinitives", "the silence"],
    "adv":  ["furiously", "deliciously", "rectangularly", "primely", "acoustically", "grammatically", "logically"],
}


def _fill_template(template: str, vocab: Dict[str, List[str]], rng: random.Random) -> str:
    """Fill placeholder slots in template with random words from vocab."""
    res = template
    for slot, words in vocab.items():
        placeholder = "{" + slot + "}"
        while placeholder in res:
            res = res.replace(placeholder, rng.choice(words), 1)
    return res


def _scramble_sentence(sentence: str, rng: random.Random) -> str:
    """Scramble word order to destroy syntactic structure."""
    words = sentence.rstrip(".").split()
    if len(words) <= 2:
        return sentence
    shuffled = list(words)
    while shuffled == words:
        rng.shuffle(shuffled)
    return " ".join(shuffled) + "."


def generate_chomsky_pairs(num_samples: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generates balanced Chomskyan diagnostic sentences:
    - Type A: Syntax Valid, Semantically Anomalous ("Colorless green ideas sleep furiously.")
    - Type B: Syntax Scrambled, Semantically Coherent lexicon
    - Type C: Syntax Valid, Semantically Coherent ("The scientist carefully examines the evidence.")
    - Type D: Syntax Scrambled, Semantically Anomalous lexicon
    """
    rng = random.Random(seed)
    records: List[Dict[str, Any]] = []
    per_type = max(1, num_samples // 4)

    for i in range(per_type):
        tmpl = rng.choice(SYNTACTIC_TEMPLATES)

        # Type C: Natural baseline (Syntax OK, Semantics OK)
        sent_c = _fill_template(tmpl, COHERENT_VOCAB, rng)

        # Type A: Chomsky original quadrant (Syntax OK, Semantics Incoherent)
        sent_a = _fill_template(tmpl, INCOHERENT_VOCAB, rng)

        # Type B: Syntactically broken word salad with coherent words
        sent_b = _scramble_sentence(sent_c, rng)

        # Type D: Completely broken (Syntax broken, Semantics Incoherent)
        sent_d = _scramble_sentence(sent_a, rng)

        suite = [
            ("A", sent_a, True, False, "Syntax Valid / Semantics Anomalous"),
            ("B", sent_b, False, True, "Syntax Scrambled / Semantics Coherent"),
            ("C", sent_c, True, True, "Syntax Valid / Semantics Coherent (Control)"),
            ("D", sent_d, False, False, "Syntax Scrambled / Semantics Anomalous (Negative Control)"),
        ]

        for type_label, sentence_text, syn_ok, sem_ok, desc in suite:
            records.append({
                "id": f"chomsky_{type_label}_{i}",
                "type": type_label,
                "sentence": sentence_text,
                "syntactic_valid": syn_ok,
                "semantic_coherent": sem_ok,
                "description": desc,
            })

    return records
