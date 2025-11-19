import torch
from typing import Optional
from flash_attn import flash_attn_func


def fatten_attn_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: Optional["KV_cache"] = None,
    layer_idx: int = 0,
    slot_idx: int = 0,
    is_prefill: bool = False,
    causal: bool = True
):
    if kv_cache is None:
        kv_cache.update(layer_idx, k[:, :, -1:], v[:, :, -1:], slot_idx)
    else:
        cached_k, cached_v = kv_cache.get(layer_idx, slot_idx)
        k = torch.cat([cached_k, k], dim=2)
        v = torch.cat([cached_v, v], dim=2)
        kv_cache.update(layer_idx, k[:, :, -1:], v[:, :, -1:], slot_idx)

    output = flash_attn_func(
        q, k, v,
        causal=causal,
        softmax_scale=1.0 / (q.size(-1) ** 0.5)
    )

    return output