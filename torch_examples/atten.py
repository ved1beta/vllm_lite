import math
import torch
def pytorch_flash_attention(Q, K, V, mask=None, scale=None):
    """
    PyTorch fallback implementation of Flash Attention
    """
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    # Apply mask if provided
    if mask is not None:
        # Expand mask to match attention scores shape
        mask = mask.unsqueeze(1).expand(-1, Q.shape[1], -1, -1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)
   
    output = torch.matmul(attn_weights, V)
    
    return output

