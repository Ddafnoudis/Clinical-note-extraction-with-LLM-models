"""
Defines the special tokens and the tokenization
Using those it defines the tokenizer
"""
# Tokenization library
import tiktoken
from typing import Dict


def preprocess_tokenizer(tokenizer_model: Dict)-> tiktoken.core.Encoding:
    """
    Define the tokenizer
    """
    # Create structured markets to convert input text to tokens
    special_tokens = [
    "<|begin_of_text|>",  
    "<|end_of_text|>",
    "<|reserved_special_token_0|>",  
    "<|reserved_special_token_1|>",  
    "<|reserved_special_token_2|>",  
    "<|reserved_special_token_3|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|reserved_special_token_4|>",
    "<|eot_id|>",
    "[MASK]",
    ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
    
    # Define the patterns based on which text will be break into tokens
    tokenize_breaker = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
   
   # Initialize tokenizer  with specified parameters
    tokenizer = tiktoken.Encoding(
        name=tokenizer_model,
        pat_str=tokenize_breaker,
        mergeable_ranks=tokenizer_model,
        # Set special tokens with indices
        special_tokens={token: len(tokenizer_model) + i for i, token in enumerate(special_tokens)},
    )

    return tokenizer


if __name__=="__main__":
    preprocess_tokenizer()
