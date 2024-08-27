"""
Apply SwiGLU activation funtion to the embedding after normalization
"""
import torch
from typing import Dict


def swiglu_activation_function(embedding_after_edit_normalized: torch.Tensor,
                               model: Dict):
    """
    Define the Feedforward weights
    """
    # Retrieve weights for feedforward layer
    w1 = model["layers.0.feed_forward.w1.weight"]
    w2 = model["layers.0.feed_forward.w2.weight"]
    w3 = model["layers.0.feed_forward.w3.weight"]

    # Perform operations for feedforward layer
    output_after_feedforward = torch.matmul(torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)

    # Print the shape of the resulting tensor after feedforward
    print(f"\n Shape of output after feedforward: {output_after_feedforward.shape}\n")

    return output_after_feedforward


if __name__=="__main__":
    swiglu_activation_function()
