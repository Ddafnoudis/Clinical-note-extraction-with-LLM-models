"""
Define the unnormalized embedding of tokens and normalize the embedding.
Params: 1) Model
        2) Vocabulary size
        3) Dimensions
        4) Tokens
        5) Normalized weights = model["layers.0.attention_norm.weight"]
"""
import torch
import torch.nn as nn
from typing import Dict


def embedding_layer(model: Dict, vocab_size: int, dim: int, tokens: torch.Tensor, norm_eps: int)-> torch.Tensor: 
    """
    Define the Embedding layer
    """
    # Define the embedding layer with the dimension and vocabulary size
    embedding_layer = torch.nn.Embedding(vocab_size, dim)
    # Copy pre-trained token embeddings to the embedding layer
    embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])   
    # Convert token embedding to torch.bfloat16
    token_embeddings_unnormal = embedding_layer(tokens).to(torch.bfloat16)

    # Calculating RMSNorm
    def rms_norm(tensor, norm_weights):
        """
        Applie root-mean-square (RMS) normalization to the input tensor, 
        using the models's normalization weights.
        """
        # Calculate the mean of the square of tensor values along the last dimension
        squared_mean: torch.Tensor = tensor.pow(2).mean(-1, keepdim=True)
        # Add a small value to avoid division by zero
        normalized: torch.Tensor = torch.rsqrt(squared_mean + norm_eps)
    
        # Multiply normalized tensor by the provided normalization weights
        return (tensor * normalized) * norm_weights
    
    # RMS normalization and provided normalization weights
    token_embeddings = rms_norm(token_embeddings_unnormal, 
                                norm_weights=model["layers.0.attention_norm.weight"])
    
    return token_embeddings, token_embeddings_unnormal


if __name__=="__main__":
    embedding_layer()
