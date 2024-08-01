"""
"""
import torch
from typing import Dict


def key_tensor(model: Dict, dim: int, n_kv_heads: int, token: torch.Tensor, token_embeddings: torch.Tensor):
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



if __name__=="__main__":
    key_tensor()