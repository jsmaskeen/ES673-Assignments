import os
import sys
from dataclasses import dataclass
from typing import Tuple, Optional, Any

import torch
try:
    from transformers import AutoTokenizer, AutoConfig, Gemma2ForCausalLM
except ImportError:
    AutoTokenizer, AutoConfig, Gemma2ForCausalLM = None, None, None


@dataclass
class GemmaTelemetry:
    """Architectural properties of Gemma-2."""
    model_id: str
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    hidden_size: int
    head_dim: int
    gqa_ratio: int
    intermediate_size: int
    device: str
    dtype: str

    @classmethod
    def from_model(cls, model: Any, model_id: str) -> "GemmaTelemetry":
        cfg = model.config
        num_q = cfg.num_attention_heads
        num_kv = cfg.num_key_value_heads
        return cls(
            model_id=model_id,
            num_layers=cfg.num_hidden_layers,
            num_q_heads=num_q,
            num_kv_heads=num_kv,
            hidden_size=cfg.hidden_size,
            head_dim=cfg.head_dim if hasattr(cfg, "head_dim") else cfg.hidden_size // num_q,
            gqa_ratio=num_q // num_kv,
            intermediate_size=cfg.intermediate_size,
            device=str(next(model.parameters()).device),
            dtype=str(next(model.parameters()).dtype),
        )


def get_hf_token(explicit_token: Optional[str] = None) -> Optional[str]:
    """
    Retrieve Hugging Face authentication token.
    Falls back to environment variable HF_TOKEN, or default placeholder.
    """
    if explicit_token and explicit_token != "blah blah":
        return explicit_token
    token = os.environ.get("HF_TOKEN", None)
    if token and token != "blah blah":
        return token
    return "blah blah"


def load_backbone(
    model_id: str = "google/gemma-2-2b",
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    token: Optional[str] = None,
) -> Tuple[Gemma2ForCausalLM, AutoTokenizer, GemmaTelemetry]:
    """
    Load Google Gemma-2 in evaluation mode with eager attention implementation
    for reliable hook attachment during ablation surgery and hidden state extraction.
    """
    torch.set_grad_enabled(False)

    hf_token = get_hf_token(token)

    # Determine execution device & precision
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if dtype is None:
        if device.startswith("cuda") and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif device.startswith("cuda"):
            dtype = torch.float16
        else:
            dtype = torch.float32

    if AutoTokenizer is None:
        raise ImportError("Package 'transformers' is required. Please install it with `pip install transformers`.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token if hf_token != "blah blah" else None,
            use_fast=True,
        )
    except Exception as e:
        if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e) or hf_token == "blah blah":
            print("\n" + "=" * 70, file=sys.stderr)
            print("[AUTHENTICATION ERROR] Gated Hugging Face Model Access Required", file=sys.stderr)
            print(f"To download '{model_id}', please:", file=sys.stderr)
            print("1. Accept the model license at: https://huggingface.co/google/gemma-2-2b", file=sys.stderr)
            print("2. Set your Hugging Face token in PowerShell:", file=sys.stderr)
            print("   $env:HF_TOKEN = \"your_hf_token_here\"", file=sys.stderr)
            print("=" * 70 + "\n", file=sys.stderr)
        raise e

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = Gemma2ForCausalLM.from_pretrained(
            model_id,
            token=hf_token if hf_token != "blah blah" else None,
            torch_dtype=dtype,
            device_map="auto" if device.startswith("cuda") else None,
            attn_implementation="eager",
        )
    except Exception as e:
        if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e) or hf_token == "blah blah":
            print("\n" + "=" * 70, file=sys.stderr)
            print("[AUTHENTICATION ERROR] Gated Hugging Face Model Access Required", file=sys.stderr)
            print(f"To download '{model_id}', please:", file=sys.stderr)
            print("1. Accept the model license at: https://huggingface.co/google/gemma-2-2b", file=sys.stderr)
            print("2. Set your Hugging Face token in PowerShell:", file=sys.stderr)
            print("   $env:HF_TOKEN = \"your_hf_token_here\"", file=sys.stderr)
            print("=" * 70 + "\n", file=sys.stderr)
        raise e

    model.eval()
    if not device.startswith("cuda"):
        model.to(torch.device(device))

    telemetry = GemmaTelemetry.from_model(model, model_id)
    print(f"[common_gemma.backbone] Model successfully loaded. Architecture: {telemetry.num_layers} layers, "
          f"{telemetry.num_q_heads} Q-heads, {telemetry.num_kv_heads} KV-heads (GQA ratio {telemetry.gqa_ratio}:1).")

    return model, tokenizer, telemetry
