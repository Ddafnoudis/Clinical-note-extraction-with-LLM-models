"""
Define the files that have been downloaded from Meta-Llama-3.1-8B
Params: 1) Model
        2) Tokenizer
        3) Parameters
"""
import json
import torch
from typing import Dict
from tiktoken.load import load_tiktoken_bpe


def model_config(tokenizer: str, model: str, params: str)-> Dict:
    """
    Configure model
    """
    # Load tokenizer
    tokenizer = load_tiktoken_bpe(tokenizer)
    print("Tokenizer has been loaded\n")

    # Load the model
    model = torch.load(model)
    print("Model has been loaded!\n")

    # Define the first 5 keys of the models
    twenty_models_keys = '\n'.join(list(model.keys())[:5])
    # Print the 20 keys of the model
    print(f"The first 20 keys of the model are: \n{twenty_models_keys}\n")
    # Define the last 10 keys of the model
    last_ten_model_keys = '\n'.join(list(model.keys())[-10:])
    # Print the last 10 keys of the model
    print(f"The last 10 keys of the model are: \n{last_ten_model_keys}\n")

    # Open the parameters JSON file
    with open(params, "r") as f:
        params_config = json.load(f)
    print("Parameters has been passed\n")

    return tokenizer, model, params_config


if __name__=="__main__":
    model_config()
