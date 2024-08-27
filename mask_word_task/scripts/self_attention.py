"""
Apply self attention 
"""
import torch
from typing import Dict


def self_attention_qkv(model: Dict, token_embeddings: torch.Tensor, 
                       tokens: torch.Tensor, dim: int, 
                       n_kv_heads: int, q_per_token_rotated: torch.Tensor, 
                       k_per_token_rotated: torch.Tensor):
    """
    Calculate query and key and mask tokens
    """
    # Calculate the query and key dot products per token
    qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)    
    # print(f"Shape of QK_per_token: {qk_per_token.shape}");exit()
    def mask_token():
        """
        Define the mask of the tokens
        """
        # Create a mask tensor filled with negative infinity values
        mask_inf = torch.full((len(tokens), len(tokens)), float("-inf"), device=tokens.device)
        # Set upper triangular part of the mask tensor to negative infinity
        mask = torch.triu(mask_inf, diagonal=1)
        
        return mask

    # Define the mask_token funtion
    mask = mask_token()
    # Add the mask to the query-key dot products per token
    qk_per_token_after_masking = qk_per_token + mask
     # Apply softmax along the second dimension after masking
    qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
    
    # Retrieve the value weight for the first layer of attention
    v_layer_0 = model["layers.0.attention.wv.weight"] 
    # Reshape value weight for the first layer of attention to separate heads
    v_layer_0 = v_layer_0.view(n_kv_heads, v_layer_0.shape[0] // n_kv_heads, dim)
    # Extract the value weight for the first head of the first layer of attention
    v_layer_head_0 = v_layer_0[0]
    # Calculate value per token by matrix multiplication
    v_per_token = torch.matmul(token_embeddings, v_layer_head_0.T)
    # print(v_per_token.shape)

    # Calculate the Query, Key and Value attention by matrix multiplication
    qmk_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)

    return v_layer_0


if __name__=="__main__":
    self_attention_qkv()
