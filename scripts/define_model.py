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
<<<<<<< HEAD
    model = torch.load(model, map_location=torch.device("cpu"))
=======
    model = torch.load(model, map_location=torch.device('cpu'))
>>>>>>> f6dca2dacb12fcf97916bbe3507da3170b672f4e
    print("Model has been loaded!\n")

    # Define the first 5 keys of the models
    model_attr = '\n'.join(list(model.keys()))
    # Print the 20 keys of the model
    print(f"The attributes of the model are: \n{model_attr}\n")
    

    # Open the parameters JSON file
    with open(params, "r") as f:
        params_config = json.load(f)
    print("Parameters has been passed\n")

    return tokenizer, model, params_config


if __name__=="__main__":
    model_config()
