"""
"""
import torch
from typing import Dict


def key_tensor(model: Dict, 
               dim: int, 
               n_kv_heads: int, 
               token_embeddings: torch.Tensor,
               freqs_cis: torch.Tensor):
    """
    """
    # Retrieve the weight of the attentions mechanism's query in the first layer of the model
    k_layer_0 = model["layers.0.attention.wk.weight"]
    # Reshape key weight for the first layer to separate heads
    k_layer_0 = k_layer_0.view(n_kv_heads, k_layer_0.shape[0] // n_kv_heads, dim)
    # print(k_layer_0.shape)
    
    # Define the first head from the key layer
    k_layer_head_0 = k_layer_0[0]
    
    # Calculate key per token by matric multiplication
    k_per_token = torch.matmul(token_embeddings, k_layer_head_0.T)
    # Split key per token into pairs and convert to float
    k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
    # Convert key per token to complex numbers
    k_per_token_as_complex_num = torch.view_as_complex(k_per_token_split_into_pairs)
    # Rotate complex key per token by frequencies
    k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_num * freqs_cis)
    # Reshape rotated key per token to match the original shape
    k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)

    return k_layer_0, k_per_token_rotated


if __name__=="__main__":
    key_tensor()
