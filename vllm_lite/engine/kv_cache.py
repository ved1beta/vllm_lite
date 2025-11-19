import torch
from typing import Tuple, Optional


class KVCache:
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device
        
        self.cache = torch.zeros(
            num_layers,
            2,  # 0 for key, 1 for value
            max_batch_size,
            num_heads,
            max_seq_len,
            head_dim,
            dtype=dtype,
            device=device
        )
        
        self.seq_lens = torch.zeros(max_batch_size, dtype=torch.long, device=device)
        
        self.active_slots = torch.zeros(max_batch_size, dtype=torch.bool, device=device)
    
    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_idx: int
    ) -> None:
        seq_len = key.size(1)
        start_pos = self.seq_lens[slot_idx].item()
        end_pos = start_pos + seq_len
        
        if end_pos > self.max_seq_len:
            raise ValueError(
                f"Sequence length {end_pos} exceeds maximum {self.max_seq_len}"
            )
        
        self.cache[layer_idx, 0, slot_idx, :, start_pos:end_pos, :] = key
        
        self.cache[layer_idx, 1, slot_idx, :, start_pos:end_pos, :] = value
        
        self.seq_lens[slot_idx] = end_pos
        self.active_slots[slot_idx] = True
    
    def get(
        self,
        layer_idx: int,
        slot_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = self.seq_lens[slot_idx].item()
        
        if seq_len == 0:
            return None, None
        
        key = self.cache[layer_idx, 0, slot_idx, :, :seq_len, :]
        value = self.cache[layer_idx, 1, slot_idx, :, :seq_len, :]
        
        return key, value
    
    def get_batch(
        self,
        layer_idx: int,
        slot_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size = len(slot_indices)
        max_seq_len_in_batch = self.seq_lens[slot_indices].max().item()
        
        keys = torch.zeros(
            batch_size, self.num_heads, max_seq_len_in_batch, self.head_dim,
            dtype=self.dtype, device=self.device
        )
        values = torch.zeros(
            batch_size, self.num_heads, max_seq_len_in_batch, self.head_dim,
            dtype=self.dtype, device=self.device
        )
        
        for i, slot_idx in enumerate(slot_indices):
            seq_len = self.seq_lens[slot_idx].item()
            if seq_len > 0:
                keys[i, :, :seq_len, :] = self.cache[layer_idx, 0, slot_idx, :, :seq_len, :]
                values[i, :, :seq_len, :] = self.cache[layer_idx, 1, slot_idx, :, :seq_len, :]
        
        return keys, values
    
    def clear_slot(self, slot_idx: int) -> None:
        """Clear a cache slot (when request is finished)"""
        self.seq_lens[slot_idx] = 0
        self.active_slots[slot_idx] = False
    
    def get_free_slot(self) -> Optional[int]:
        free_slots = (~self.active_slots).nonzero(as_tuple=True)[0]
        if len(free_slots) == 0:
            return None
        return free_slots[0].item()
    
    def get_seq_len(self, slot_idx: int) -> int:
        return self.seq_lens[slot_idx].item()

