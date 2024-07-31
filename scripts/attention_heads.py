"""
"""
import torch
from typing import Dict

def query_key_value(model: Dict, n_heads:int, dim: int, token: torch.Tensor, token_embeddings: torch.Tensor):
    """
    """
    # Retrieve the query of the first attention layer
    q_layer_0 = model["layers.0.attention.wq.weight"]
    # Calculate the dimension per head
    head_dim = q_layer_0.shape[0] // n_heads
    # Reshape the query
    q_layer_0 = q_layer_0.view(n_heads, head_dim, dim)
    #print(q_layer_0.shape)
    # Extract the query weight
    q_layer_head_0 = q_layer_0[0]
    # Ensure all tensors are of the same dtype
    token = token.to(q_layer_0.dtype)
    token_embeddings = token_embeddings.to(q_layer_0.dtype)
    q_layer_head_0 = q_layer_head_0.to(q_layer_0.dtype)
    # Perform a matrix multiplication
    q_per_token = torch.matmul(token, torch.matmul(token_embeddings, q_layer_head_0.T))
    print(type(q_per_token),"\n")
    print(q_per_token.shape)


if __name__=="__main__":
    query_key_value()
