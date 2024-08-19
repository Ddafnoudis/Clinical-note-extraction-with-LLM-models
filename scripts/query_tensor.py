"""
"""
import torch
from typing import Dict


def query_tensor(model: Dict, n_heads:int, dim: int, token: torch.Tensor, token_embeddings: torch.Tensor, rope_theta: float):
    """
    """
    ####################### QUERY #######################
    # Retrieve the weight of the attentions mechanism's query in the first layer of the model
    q_layer_0 = model["layers.0.attention.wq.weight"]
    # Calculate the dimension per headgit 
    head_dim = q_layer_0.shape[0] // n_heads
    # Reshape the query
    q_layer_0 = q_layer_0.view(n_heads, head_dim, dim)
    # Extract the query weight
    q_layer_head_0 = q_layer_0[0]
    # Ensure all tensors are of the same dtype
    token = token.to(q_layer_0.dtype)
    # print(f"The type of the token is: {type(token)}")
    token_embeddings = token_embeddings.to(q_layer_0.dtype)
    q_layer_head_0 = q_layer_head_0.to(q_layer_0.dtype)
    # Perform a matrix multiplication
    q_per_token = torch.matmul(token_embeddings, q_layer_head_0.T)
    # Convert queries per token to float and split into pairs
    q_per_t_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
    # Generate values from zero to one and split into 64 parts
    zero_to_one_split_into_64_parts = torch.tensor(range(64))/64
    # Calculate frequencies using a power operation (1-D for the vecs2)
    frequencies = 1.0 / (rope_theta ** zero_to_one_split_into_64_parts)
    # Convert queries per token to complex numbers
    q_per_token_as_compl_num = torch.view_as_complex(q_per_t_split_into_pairs)
    # Calculate frequencies for each token using outer product of arange(17) and freqs
    freqs_for_each_token = torch.outer(torch.arange(len(q_per_token_as_compl_num)), vec2=frequencies)
    # Calculate complex numbers 
    freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)
    q_per_token_as_complex_num_rotated = q_per_token_as_compl_num * freqs_cis
    
    # Convert rotated complex numbers back to real numbers
    q_per_t_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_num_rotated)
    # print(q_per_t_split_into_pairs_rotated.shape)
    q_per_token_rotated = q_per_t_split_into_pairs_rotated.view(q_per_token.shape)
    # print(q_per_token_rotated.shape)

    return q_layer_0, q_per_token_rotated, freqs_cis


if __name__=="__main__":
    query_tensor()
