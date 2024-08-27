"""
Define the parameters of the model
"""
import torch
from typing import Dict, Any


def param_configuration(config: Dict)-> Any:
    """
    Define the parameters
    """
    # Store the values from the parameters of the model
    # Dimension
    dim = config["dim"]
    # Layers
    n_layers = config["n_layers"]
    # Heads
    n_heads = config["n_heads"]
    # KV_heads
    n_kv_heads = config["n_kv_heads"]
    # Vocabulary
    vocab_size = config["vocab_size"]
    # Multiple
    multiple_of = config["multiple_of"]
    # Multiplier
    ffn_dim_multiplier = config["ffn_dim_multiplier"]
    # Epsilon
    norm_eps = config["norm_eps"]
    # RoPE
    rope_theta = torch.tensor(config["rope_theta"])

    return dim, n_layers, n_heads, n_kv_heads, vocab_size, multiple_of, ffn_dim_multiplier, norm_eps, rope_theta


if __name__=="__main__":
    param_configuration()
