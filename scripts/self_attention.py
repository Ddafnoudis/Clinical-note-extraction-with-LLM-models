"""
"""
import torch
from typing import Dict


def self_attention_qk(tokens: torch.Tensor, q_per_token_rotated: torch.Tensor, k_per_token_rotated: torch.Tensor):
    """
    Calculate query and key and mask tokens
    """
    # Calculate the query and key dot products per token
    qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)    

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
    print(qk_per_token_after_masking_after_softmax)



if __name__=="__main__":
    self_attention_qk()
