from typing import List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
try:
    from transformers import PreTrainedTokenizerBase
except ImportError:
    PreTrainedTokenizerBase = Any


@torch.no_grad()
def extract_all_layer_hidden_states(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """
    Executes forward pass with output_hidden_states=True and returns
    a list of 27 hidden state tensors (Layer 0 = embedding, Layers 1..26 = decoder layers).
    Each tensor has shape [batch_size, seq_len, hidden_size].
    """
    model.eval()
    try:
        device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        device = torch.device("cpu")
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    # outputs.hidden_states is a tuple of (num_layers + 1) tensors
    return [h.detach() for h in outputs.hidden_states]


@torch.no_grad()
def get_sentence_embeddings_all_layers(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    pooling: str = "mean",
    max_length: int = 512,
) -> List[torch.Tensor]:
    """
    Returns 27 1D vectors (CPU float32), each representing the entire sentence
    at a specific layer stage (0 to 26).
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", None)

    hidden_states = extract_all_layer_hidden_states(model, input_ids, attention_mask)

    pooled_layers: List[torch.Tensor] = []
    for h in hidden_states:
        # h: [1, seq_len, hidden_size]
        if pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(h.device)
                sum_vec = (h * mask).sum(dim=1)
                lens = mask.sum(dim=1).clamp(min=1)
                vec = sum_vec / lens
            else:
                vec = h.mean(dim=1)
        elif pooling == "last":
            vec = h[:, -1, :]
        else:
            vec = h.mean(dim=1)

        pooled_layers.append(vec.squeeze(0).to(torch.float32).cpu())

    return pooled_layers


@torch.no_grad()
def get_token_embeddings_all_layers(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    sentence: str,
    target_word: str,
    char_start: Optional[int] = None,
    char_end: Optional[int] = None,
) -> Optional[List[torch.Tensor]]:
    """
    Extracts the contextual token representation of a specific target word
    across all 27 layers (0 to 26).
    If char_start/end are provided, aligns tokens to character spans.
    Otherwise searches for matching word tokens.
    """
    enc = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True)
    input_ids = enc["input_ids"]
    offset_mapping = enc.get("offset_mapping", None)

    target_token_indices = []

    if offset_mapping is not None and char_start is not None and char_end is not None:
        offsets = offset_mapping[0].tolist()
        for idx, (s, e) in enumerate(offsets):
            if s >= char_start and e <= char_end and s < e:
                target_token_indices.append(idx)
            elif (s < char_end and e > char_start) and s < e:
                target_token_indices.append(idx)

    # Fallback to token string matching if offset mapping alignment failed
    if not target_token_indices:
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        clean_word = target_word.lower().strip()
        for idx, tok in enumerate(tokens):
            clean_tok = tok.replace(" ", "").replace("Ġ", "").lower()
            if clean_tok == clean_word or (len(clean_tok) > 2 and clean_tok in clean_word):
                target_token_indices.append(idx)

    if not target_token_indices:
        # Default to middle token if match fails
        target_token_indices = [input_ids.size(1) // 2]

    hidden_states = extract_all_layer_hidden_states(model, input_ids)

    layer_word_vectors: List[torch.Tensor] = []
    for h in hidden_states:
        # Average over sub-tokens if target_word was split into multiple tokens
        sub_vecs = h[0, target_token_indices, :]  # [num_subtokens, hidden]
        word_vec = sub_vecs.mean(dim=0)
        layer_word_vectors.append(word_vec.to(torch.float32).cpu())

    return layer_word_vectors
