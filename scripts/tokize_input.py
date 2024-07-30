import torch
import tiktoken
from typing import List


def tokenize_input(tokenizer: tiktoken.core.Encoding, train_text: List[str])-> torch.Tensor:
    """
    From the text encode tokens and return a tensor
    """
    # Iterate over the train text
    for text in train_text:
        # Enocde the train text and first use a special token
        tokens = [128000] + tokenizer.encode(text)
        # Convert the list of tokens into a PyTorch tensor
        tokens = torch.tensor(tokens)
        # Decode each token back into its corresponding string for you to see
        prompt_split_token = [tokenizer.decode([token.item()]) for token in tokens]
        # print(prompt_split_token)

    return tokens

if __name__=="__main__":
    tokenize_input()
