import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn


@dataclass
class SurgerySpec:
    """
    Specification for ablation surgery on Gemma-2.
    """
    surgery_type: str = "none"  # 'none' | 'heads' | 'layers' | 'emb_mask' | 'emb_spectral'
    severity: float = 0.0  # fraction in [0.0, 1.0]
    head_mode: str = "q_only"  # 'q_only' | 'kv_group' | 'full_head'
    component: str = "block"  # for layers: 'attn' | 'mlp' | 'block'
    position: str = "random"  # 'early' | 'middle' | 'late' | 'random' | 'consecutive'
    perturbation_type: str = "zero"  # 'zero' | 'gaussian_noise' | 'mean_clamp'
    seed: int = 42
    metadata: Dict[str, Any] = field(default_factory=dict)


def select_indices_random(total: int, k: int, seed: int) -> List[int]:
    """Deterministically select k random indices from total."""
    if k <= 0:
        return []
    if k >= total:
        return list(range(total))
    rng = random.Random(seed)
    idxs = list(range(total))
    rng.shuffle(idxs)
    return sorted(idxs[:k])


def select_indices_consecutive(total: int, k: int, start: int = 0) -> List[int]:
    """Select k consecutive indices starting at start."""
    if k <= 0:
        return []
    end = min(start + k, total)
    return list(range(start, end))


def plan_head_selection(
    num_q_heads: int,
    num_kv_heads: int,
    head_mode: str,
    severity: float,
    policy: str,
    seed: int,
) -> Dict[str, Any]:
    """
    Plan head selection according to GQA modes.
    Returns targeted Query heads and Key/Value groups.
    """
    gqa_ratio = num_q_heads // num_kv_heads

    if head_mode == "q_only":
        k_q = int(round(num_q_heads * severity))
        if policy == "random":
            q_selected = select_indices_random(num_q_heads, k_q, seed)
        else:
            q_selected = select_indices_consecutive(num_q_heads, k_q, start=0)
        return {"q_heads": q_selected, "kv_groups": []}

    elif head_mode == "kv_group":
        k_kv = int(round(num_kv_heads * severity))
        if policy == "random":
            kv_selected = select_indices_random(num_kv_heads, k_kv, seed)
        else:
            kv_selected = select_indices_consecutive(num_kv_heads, k_kv, start=0)
        associated_q = []
        for g in kv_selected:
            associated_q.extend(list(range(g * gqa_ratio, (g + 1) * gqa_ratio)))
        return {"q_heads": [], "kv_groups": kv_selected, "associated_q_heads": sorted(associated_q)}

    elif head_mode == "full_head":
        k_kv = int(round(num_kv_heads * severity))
        if policy == "random":
            kv_selected = select_indices_random(num_kv_heads, k_kv, seed)
        else:
            kv_selected = select_indices_consecutive(num_kv_heads, k_kv, start=0)
        q_selected = []
        for g in kv_selected:
            q_selected.extend(list(range(g * gqa_ratio, (g + 1) * gqa_ratio)))
        return {"q_heads": q_selected, "kv_groups": kv_selected}

    else:
        raise ValueError(f"Unknown head_mode: {head_mode}")


def plan_layer_selection(
    num_layers: int,
    severity: float,
    position: str,
    seed: int,
) -> List[int]:
    """
    Select layer indices based on position policy.
    """
    k = int(round(num_layers * severity))
    if k <= 0:
        return []
    if k >= num_layers:
        return list(range(num_layers))

    if position == "early":
        return list(range(0, k))
    elif position == "late":
        return list(range(num_layers - k, num_layers))
    elif position == "middle":
        start = max(0, (num_layers - k) // 2)
        return list(range(start, start + k))
    elif position == "random":
        return select_indices_random(num_layers, k, seed)
    elif position == "consecutive":
        return select_indices_consecutive(num_layers, k, start=0)
    else:
        raise ValueError(f"Unknown position policy: {position}")


def apply_perturbation(tensor: torch.Tensor, perturbation_type: str = "zero") -> torch.Tensor:
    """
    Applies the specified perturbation to the tensor:
    - 'zero': Clamps activations to 0.0 (classical lesion / synaptic necrosis).
    - 'gaussian_noise': Injects Gaussian noise matching activation variance (synaptic jitter).
    - 'mean_clamp': Clamps to spatial/channel mean (eliminates dynamic entropy while preserving DC excitation).
    """
    if perturbation_type in ("zero", None, "none"):
        return torch.zeros_like(tensor)
    elif perturbation_type == "gaussian_noise":
        std = tensor.std()
        mean = tensor.mean()
        if std == 0.0 or torch.isnan(std):
            std = 0.02
        return torch.randn_like(tensor) * std + mean
    elif perturbation_type == "mean_clamp":
        mean = tensor.mean(dim=-1, keepdim=True)
        return mean.expand_as(tensor)
    else:
        raise ValueError(f"Unknown perturbation_type: {perturbation_type}. Supported: 'zero', 'gaussian_noise', 'mean_clamp'.")


def make_q_head_pre_hook(head_dim: int, q_heads_to_zero: List[int], perturbation_type: str = "zero"):
    slices = [(h * head_dim, (h + 1) * head_dim) for h in q_heads_to_zero]

    def hook(module, args):
        inp = args[0]
        if not slices or inp is None:
            return args
        modified = inp.clone()
        for s, e in slices:
            modified[..., s:e] = apply_perturbation(modified[..., s:e], perturbation_type)
        return (modified,) + args[1:]

    return hook


def make_kv_group_hook(head_dim: int, kv_groups_to_zero: List[int], perturbation_type: str = "zero"):
    slices = [(g * head_dim, (g + 1) * head_dim) for g in kv_groups_to_zero]

    def hook(module, inp, output):
        if not slices or output is None:
            return output
        out = output.clone()
        for s, e in slices:
            out[..., s:e] = apply_perturbation(out[..., s:e], perturbation_type)
        return out

    return hook


def make_zero_output_hook(perturbation_type: str = "zero"):
    def hook(module, inp, output):
        if isinstance(output, tuple):
            return (apply_perturbation(output[0], perturbation_type),) + output[1:]
        return apply_perturbation(output, perturbation_type)
    return hook


def make_skip_layer_hook():
    def hook(module, args, output):
        inp = args[0]
        if isinstance(output, tuple):
            return (inp,) + output[1:]
        return inp
    return hook


def make_embedding_mask_hook(frac: float, perturbation_type: str = "zero"):
    def hook(module, inp, output):
        hidden_size = output.size(-1)
        k = int(round(hidden_size * frac))
        if k <= 0:
            return output
        out = output.clone()
        out[..., :k] = apply_perturbation(out[..., :k], perturbation_type)
        return out
    return hook


@contextmanager
def apply_surgery(model: nn.Module, spec: SurgerySpec):
    """
    Context manager that installs ablation surgery hooks on model and
    guarantees complete detachment upon exit.
    """
    handles = []
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_q_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim if hasattr(cfg, "head_dim") else cfg.hidden_size // num_q_heads

    try:
        if spec.surgery_type in ("none", None) or spec.severity <= 0.0:
            spec.metadata["active_modifications"] = "none"
            yield handles
            return

        elif spec.surgery_type == "heads":
            plan = plan_head_selection(
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_mode=spec.head_mode,
                severity=spec.severity,
                policy=spec.position,
                seed=spec.seed,
            )
            spec.metadata["head_plan"] = plan

            q_heads = plan.get("q_heads", [])
            kv_groups = plan.get("kv_groups", [])

            for layer_idx in range(num_layers):
                attn = model.model.layers[layer_idx].self_attn

                if q_heads:
                    h = attn.o_proj.register_forward_pre_hook(
                        make_q_head_pre_hook(head_dim, q_heads, perturbation_type=spec.perturbation_type)
                    )
                    handles.append(h)

                if kv_groups:
                    h_k = attn.k_proj.register_forward_hook(
                        make_kv_group_hook(head_dim, kv_groups, perturbation_type=spec.perturbation_type)
                    )
                    h_v = attn.v_proj.register_forward_hook(
                        make_kv_group_hook(head_dim, kv_groups, perturbation_type=spec.perturbation_type)
                    )
                    handles.extend([h_k, h_v])

        elif spec.surgery_type == "layers":
            layers_to_ablate = plan_layer_selection(
                num_layers=num_layers,
                severity=spec.severity,
                position=spec.position,
                seed=spec.seed,
            )
            spec.metadata["layers_ablated"] = layers_to_ablate
            comp = spec.component.lower()

            for layer_idx in layers_to_ablate:
                layer = model.model.layers[layer_idx]
                if comp == "attn":
                    h = layer.self_attn.o_proj.register_forward_hook(
                        make_zero_output_hook(perturbation_type=spec.perturbation_type)
                    )
                    handles.append(h)
                elif comp == "mlp":
                    h = layer.mlp.down_proj.register_forward_hook(
                        make_zero_output_hook(perturbation_type=spec.perturbation_type)
                    )
                    handles.append(h)
                elif comp == "block":
                    h = layer.register_forward_hook(make_skip_layer_hook())
                    handles.append(h)
                else:
                    raise ValueError(f"Unknown layer component: {comp}")

        elif spec.surgery_type == "emb_mask":
            spec.metadata["masked_fraction"] = spec.severity
            h = model.model.embed_tokens.register_forward_hook(
                make_embedding_mask_hook(spec.severity, perturbation_type=spec.perturbation_type)
            )
            handles.append(h)

        else:
            raise ValueError(f"Unknown surgery_type: {spec.surgery_type}")

        yield handles

    finally:
        for handle in handles:
            handle.remove()
