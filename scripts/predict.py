"""
"""
import torch
from typing import List, Dict


def predict_masked_tokens(mask_indices: List[int], 
                          model: Dict, 
                          tokenizer: torch.Tensor, 
                          masked_clinical_notes: List[str]) -> List[str]:
    """
    Predicts the token for each [MASK] token in the input text.
    """
    predictions = []

    with torch.no_grad():
        for idx in mask_indices:
            output_logits = model(mask_indices)
            predicted_token_id = torch.argmax(output_logits, dim=-1).item()
            predicted_token = tokenizer.decode(predicted_token_id)
            predictions.append(predicted_token)

    return predictions



if __name__=="__main__":
    predict_masked_tokens()