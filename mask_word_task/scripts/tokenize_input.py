"""
Tokenize only the train text and convert it into a tensor.
"""
import torch
import tiktoken
from typing import List


def tokenize_input(tokenizer: tiktoken.core.Encoding, train_text: List[str])-> torch.Tensor:
    """
    From the text encode tokens and return a tensor
    """
    # Iterate over the train text
    for text in train_text:
        # Enocde the train text and first use a special token. Allow the "[MASK]" special character 
        tokens = [128000] + tokenizer.encode(text, allowed_special={"[MASK]"})
        # Convert the list of tokens into a PyTorch tensor
        tokens = torch.tensor(tokens)
        # Decode each token back to string (Need to be in list)
        tokens_decoded = tokenizer.decode(tokens.tolist())

    return tokens


if __name__=="__main__":
    tokenize_input()
