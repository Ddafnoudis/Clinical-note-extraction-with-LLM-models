"""
"""
import torch
from typing import Dict


def multi_head_attention(model: Dict, n_heads: int, token_embeddings: torch.Tensor,
                         q_layer_0: torch.Tensor, k_layer_0: torch.Tensor,
                         v_layer_0: torch.Tensor, tokens: torch.Tensor,
                         freqs_cis: torch.Tensor, token_embeddings_unnormal: torch.Tensor,
                         norm_eps: int):
    """
    """
    # Create an empty list to store Query, Key, Value attention of each head
    qkv_attention_list = []

    # Iterate through each head
    for head in range(n_heads):
        # Extract query, key, and value weights for the current head
        q_layer0_head = q_layer_0[head]
        # Key weights are shared across 4 heads
        k_layer0_head = k_layer_0[head // 4]
        # Value weights are shared across 4 heads  
        v_layer0_head = v_layer_0[head // 4]  
        
        # Calculate query per token by matrix multiplication
        q_per_token = torch.matmul(token_embeddings, q_layer0_head.T)
        
        # Calculate key per token by matrix multiplication
        k_per_token = torch.matmul(token_embeddings, k_layer0_head.T)
        
        # Calculate value per token by matrix multiplication
        v_per_token = torch.matmul(token_embeddings, v_layer0_head.T)
        
        # Split query per token into pairs and rotate them
        q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
        q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
        q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis[:len(tokens)])
        q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)
        
        # Split key per token into pairs and rotate them
        k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
        k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
        k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis[:len(tokens)])
        k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)
        
        # Calculate query-key dot products per token
        qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T) / (128) ** 0.5
        
        # Create a mask tensor filled with negative infinity values
        mask = torch.full((len(tokens), len(tokens)), float("-inf"), device=tokens.device)
        # Set upper triangular part of the mask tensor to negative infinity
        mask = torch.triu(mask, diagonal=1)
        # Add the mask to the query-key dot products per token
        qk_per_token_after_masking = qk_per_token + mask
        
        # Apply softmax along the second dimension after masking
        qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
        
        # Calculate QKV attention by matrix multiplication
        qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
        
        # Store QKV attention for the current head
        qkv_attention_list.append(qkv_attention)

    # Print the number of QKV attentions stored
    print(len(qkv_attention_list))

    # Concatenate QKV attentions from all heads along the last dimension
    stacked_qkv_attention = torch.cat(qkv_attention_list, dim=-1)

    # Print the shape of the resulting tensor
    print(f"\n The shape of stached_qkv_attention is: {stacked_qkv_attention.shape}\n")

    # Calculate the embedding delta by matrix multiplication with the output weight
    embedding_delta = torch.matmul(stacked_qkv_attention, model["layers.0.attention.wo.weight"].T)

    # Print the shape of the resulting tensor
    print(f"\n The shape of embedding delta is: {embedding_delta.shape}\n")

    # Add the embedding delta to the unnormalized token embeddings to get the final embeddings
    embedding_after_edit = token_embeddings_unnormal + embedding_delta

    # Print the shape of the resulting tensor
    print(f"The shape of embedding after edit is : {embedding_after_edit.shape}\n")

    # Calculating RMSNorm
    def rms_norm(tensor, norm_weights):
        """
        Applie root-mean-square (RMS) normalization
        """
        # Calculate the mean of the square of tensor values along the last dimension
        squared_mean: torch.Tensor = tensor.pow(2).mean(-1, keepdim=True)
        # Add a small value to avoid division by zero
        normalized: torch.Tensor = torch.rsqrt(squared_mean + norm_eps)
    
        # Multiply normalized tensor by the provided normalization weights
        return (tensor * normalized) * norm_weights

    # Normalize edited embeddings using root mean square normalization and provided weights
    embedding_after_edit_normalized = rms_norm(embedding_after_edit, 
                                               norm_weights=model["layers.0.ffn_norm.weight"])

    # Print the shape of resulting normalized embeddings
    print(f"Shape of embedding after edit normalized: {embedding_after_edit_normalized.shape}\n")

    return embedding_after_edit_normalized


if __name__=="__main__":
    multi_head_attention()
