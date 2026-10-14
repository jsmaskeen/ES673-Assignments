import unittest
from types import SimpleNamespace
import sys
import os

import torch
import torch.nn as nn

# Ensure path to repo root and q3 root
_curr = os.path.abspath(os.path.dirname(__file__))
_repo_root = None
_q3_root = None
while _curr:
    if os.path.exists(os.path.join(_curr, "common_gemma")):
        _repo_root = _curr
    if os.path.exists(os.path.join(_curr, "synsem")) and os.path.exists(os.path.join(_curr, "experiments")):
        _q3_root = _curr
    _parent = os.path.dirname(_curr)
    if _parent == _curr:
        break
    _curr = _parent

for _p in [_repo_root, _q3_root]:
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from common_gemma.prober import extract_all_layer_hidden_states, get_sentence_embeddings_all_layers
try:
    from synsem.datasets.chomsky import generate_chomsky_pairs
    from synsem.measures.geometry import anisotropy_corrected_cosine, compute_rsm
    from synsem.measures.preference import compute_cohens_d
except ImportError:
    from q3_synsem_gemma.synsem.datasets.chomsky import generate_chomsky_pairs
    from q3_synsem_gemma.synsem.measures.geometry import anisotropy_corrected_cosine, compute_rsm
    from q3_synsem_gemma.synsem.measures.preference import compute_cohens_d


class DummyInner(nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(num_layers)])
        self.embed_tokens = nn.Embedding(50, 64)


class DummyGemmaModelForProbing(nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=num_layers,
            num_attention_heads=8,
            num_key_value_heads=4,
            hidden_size=64,
            head_dim=8,
        )
        self.model = DummyInner(num_layers=num_layers)

    def forward(self, input_ids, output_hidden_states=False, **kwargs):
        x = self.model.embed_tokens(input_ids)
        all_hidden = [x]
        for layer in self.model.layers:
            x = layer(x)
            all_hidden.append(x)

        res = SimpleNamespace(logits=torch.randn(input_ids.size(0), input_ids.size(1), 50))
        if output_hidden_states:
            res.hidden_states = tuple(all_hidden)
        return res


class TestQ3Components(unittest.TestCase):
    def setUp(self):
        self.model = DummyGemmaModelForProbing(num_layers=4)
        self.model.eval()

    def test_prober_layer_extraction(self):
        input_ids = torch.tensor([[1, 2, 3, 4]])
        layers = extract_all_layer_hidden_states(self.model, input_ids)
        # 1 embedding + 4 decoder layers = 5 layers
        self.assertEqual(len(layers), 5)
        for h in layers:
            self.assertEqual(h.shape, (1, 4, 64))

    def test_chomsky_generator(self):
        pairs = generate_chomsky_pairs(num_samples=40, seed=42)
        self.assertEqual(len(pairs), 40)
        types = set(p["type"] for p in pairs)
        self.assertEqual(types, {"A", "B", "C", "D"})
        # Check balance
        counts = {t: sum(1 for p in pairs if p["type"] == t) for t in types}
        self.assertEqual(counts["A"], 10)
        self.assertEqual(counts["B"], 10)
        self.assertEqual(counts["C"], 10)
        self.assertEqual(counts["D"], 10)

    def test_geometry_and_rsm(self):
        u = torch.tensor([1.0, 0.0, 0.0])
        v = torch.tensor([0.0, 1.0, 0.0])
        cos = anisotropy_corrected_cosine(u, v)
        self.assertAlmostEqual(cos, 0.0, places=5)

        # Same vector
        cos_self = anisotropy_corrected_cosine(u, u)
        self.assertAlmostEqual(cos_self, 1.0, places=5)

        # 3x3 RSM
        vecs = [u, v, u]
        rsm = compute_rsm(vecs)
        self.assertEqual(rsm.shape, (3, 3))
        self.assertAlmostEqual(float(rsm[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(rsm[0, 2]), 1.0, places=5)

    def test_cohens_d(self):
        margins = [1.0, 1.2, 0.8, 1.1, 0.9]
        d = compute_cohens_d(margins)
        self.assertGreater(d, 3.0, "Consistent positive margins should yield large positive Cohen's d")


if __name__ == "__main__":
    unittest.main()
