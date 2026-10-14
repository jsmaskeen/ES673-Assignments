import unittest
import torch
import torch.nn as nn
from types import SimpleNamespace

from gemma.surgery import SurgerySpec, apply_surgery, plan_head_selection, plan_layer_selection
from gemma.backbone import inspect_gqa_layout, summarize_model_parameters


class DummySelfAttention(nn.Module):
    def __init__(self, num_q_heads=8, num_kv_heads=4, head_dim=256, hidden_size=2304):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q_heads * head_dim, hidden_size, bias=False)

    def forward(self, x):
        # x: [bsz, seq, hidden]
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Dummy projection input
        attn_out = torch.ones(x.size(0), x.size(1), self.num_q_heads * self.head_dim, device=x.device)
        return self.o_proj(attn_out)


class DummyDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DummySelfAttention()
        self.mlp = nn.Sequential(nn.Linear(2304, 2304))

    def forward(self, x):
        return x + self.self_attn(x) + self.mlp(x)


class DummyInnerModel(nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([DummyDecoderLayer() for _ in range(num_layers)])
        self.embed_tokens = nn.Embedding(100, 2304)


class DummyGemmaModel(nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=num_layers,
            num_attention_heads=8,
            num_key_value_heads=4,
            hidden_size=2304,
            head_dim=256,
            intermediate_size=9216,
        )
        self.model = DummyInnerModel(num_layers)

    def forward(self, input_ids):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        return x


class TestGemmaSurgery(unittest.TestCase):
    def setUp(self):
        self.model = DummyGemmaModel(num_layers=4)
        self.model.eval()
        self.input_ids = torch.tensor([[1, 2, 3, 4]])

    def test_q_head_surgery(self):
        baseline = self.model(self.input_ids)
        spec = SurgerySpec(surgery_type="heads", severity=0.25, head_mode="q_only", seed=42)
        with apply_surgery(self.model, spec):
            ablated = self.model(self.input_ids)
            diff = (baseline - ablated).abs().max().item()
            self.assertGreater(diff, 1e-4, "Q-head ablation should alter outputs")

        # Verify restoration
        restored = self.model(self.input_ids)
        self.assertEqual((baseline - restored).abs().max().item(), 0.0, "Hooks should be removed")

    def test_kv_group_surgery(self):
        baseline = self.model(self.input_ids)
        spec = SurgerySpec(surgery_type="heads", severity=0.5, head_mode="kv_group", seed=42)
        with apply_surgery(self.model, spec):
            self.assertEqual(len(spec.metadata["head_plan"]["kv_groups"]), 2)

    def test_layer_position_surgery(self):
        baseline = self.model(self.input_ids)
        spec = SurgerySpec(surgery_type="layers", severity=0.5, component="block", position="early", seed=42)
        with apply_surgery(self.model, spec):
            ablated = self.model(self.input_ids)
            self.assertEqual(spec.metadata["layers_ablated"], [0, 1])

        restored = self.model(self.input_ids)
        self.assertEqual((baseline - restored).abs().max().item(), 0.0)

    def test_embedding_masking(self):
        baseline = self.model(self.input_ids)
        spec = SurgerySpec(surgery_type="emb_mask", severity=0.5, seed=42)
        with apply_surgery(self.model, spec):
            ablated = self.model(self.input_ids)
            diff = (baseline - ablated).abs().max().item()
            self.assertGreater(diff, 1e-4)

        restored = self.model(self.input_ids)
        self.assertEqual((baseline - restored).abs().max().item(), 0.0)

    def test_spectral_pca_masking(self):
        baseline = self.model(self.input_ids)
        spec = SurgerySpec(surgery_type="emb_spectral", severity=0.25, seed=42)
        with apply_surgery(self.model, spec):
            ablated = self.model(self.input_ids)
            diff = (baseline - ablated).abs().max().item()
            self.assertGreater(diff, 1e-4, "Spectral PCA masking should alter model output")
            self.assertIn("spectral_k", spec.metadata)
            self.assertEqual(spec.metadata["spectral_k"], int(round(2304 * 0.25)))

        restored = self.model(self.input_ids)
        self.assertEqual((baseline - restored).abs().max().item(), 0.0, "Hooks should be cleanly restored")

    def test_inspect_gqa_layout(self):
        layout = inspect_gqa_layout(self.model)
        self.assertEqual(layout["num_query_heads"], 8)
        self.assertEqual(layout["num_key_value_heads"], 4)
        self.assertEqual(layout["gqa_ratio"], 2)
        self.assertEqual(layout["kv_to_query_mapping"][0], [0, 1])
        self.assertEqual(layout["kv_to_query_mapping"][3], [6, 7])

    def test_summarize_model_parameters(self):
        summary = summarize_model_parameters(self.model)
        self.assertIn("total_parameters", summary)
        self.assertIn("embedding_parameters", summary)
        self.assertIn("attention_parameters", summary)
        self.assertIn("mlp_parameters", summary)
        self.assertGreater(summary["total_parameters"], 0)

    def test_gaussian_noise_perturbation(self):
        baseline = self.model(self.input_ids)
        spec = SurgerySpec(surgery_type="heads", severity=0.25, perturbation_type="gaussian_noise", seed=42)
        with apply_surgery(self.model, spec):
            ablated = self.model(self.input_ids)
            diff = (baseline - ablated).abs().max().item()
            self.assertGreater(diff, 1e-4, "Gaussian noise perturbation should modify outputs")
        restored = self.model(self.input_ids)
        self.assertEqual((baseline - restored).abs().max().item(), 0.0)

    def test_mean_clamp_perturbation(self):
        baseline = self.model(self.input_ids)
        spec = SurgerySpec(surgery_type="layers", severity=0.5, component="attn", perturbation_type="mean_clamp", seed=42)
        with apply_surgery(self.model, spec):
            ablated = self.model(self.input_ids)
            diff = (baseline - ablated).abs().max().item()
            self.assertGreater(diff, 1e-4, "Mean clamp perturbation should modify outputs")
        restored = self.model(self.input_ids)
        self.assertEqual((baseline - restored).abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
