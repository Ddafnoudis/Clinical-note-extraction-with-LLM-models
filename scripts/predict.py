"""
"""
import torch
from typing import Dict, List


def make_masked_predictions(masked_test_set: List[str], model: Dict, 
                            tokenizer: torch.Tensor, tokens: torch.Tensor,
                            final):
    """
    Predict Masked clinical text data 
    """
    # Create empty list
    predictions = []
    # Process each tokenized input
    for token_ids in tokens:
        print(f"TOKEN IDS: \n {token_ids}\n")
        # Make predictions for masked tokens
        with torch.no_grad():
            """
            Pass the input token IDs to the model and get the model outputs.
            """
            outputs = model["output.weight"]
            logits = outputs.logit
            print(f"Logit: \n{logits}\n")
        
        print(masked_test_set);exit()

    return predictions


if __name__=="__main__":
    make_masked_predictions()

# """
# """
# import torch
# from typing import Dict, List


# def make_masked_predictions(masked_test_set: List[str], model: Dict, 
#                             tokenizer: torch.Tensor, tokens: torch.Tensor, 
#                             final_embedding: torch.Tensor
#                             ):
#     """
#     Predict Masked clinical text data 
#     """
#     # Process each tokenized input
#     predictions = []
#     masked_token_ids = []
#     for token_ids in tokens:
#         print(f"TOKEN IDS: \n {token_ids}\n")
#         # Make predictions for masked tokens
#         with torch.no_grad():
#             outputs = model(input_ids=token_ids)
#             logits = outputs.logits
#             print(f"Logits: \n{logits}\n")
        
#         # Find the positions of masked tokens        
#         mask_positions = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero(as_tuple=True)
        
#         # Predict tokens for masked positions
#         predicted_tokens = []
#         for pos in mask_positions[0]:
#             logits_at_pos = logits[0, pos].cpu()  # Get logits for the masked position
#             predicted_token_id = torch.argmax(logits_at_pos).item()
#             predicted_token = tokenizer.decode([predicted_token_id])
#             predicted_tokens.append(predicted_token)
#             masked_token_ids.append(predicted_token_id)
        
#         # Join the predicted tokens with the original text
#         text_with_predictions = tokenizer.decode(token_ids.squeeze())
#         for pos, token in zip(mask_positions[0], predicted_tokens):
#             text_with_predictions = text_with_predictions.replace(tokenizer.mask_token, token, 1)
        
#         predictions.append(text_with_predictions)

#     return predictions, masked_token_ids


# if __name__=="__main__":
#     make_masked_predictions()
