import os
import random
import numpy as np
import torch
from typing import List, Any, Dict, Optional
from pandas import DataFrame
from sklearn.model_selection import train_test_split


def preprocess_text_dataset(train_text: List[str]):
    preprocessed_data = [line.strip() for line in train_text]

    return preprocessed_data

# Step 2: Load the tokenizer
class CustomTokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as file:
            self.vocab = file.read().splitlines()
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.mask_token = '[MASK]'
        self.mask_token_id = self.token_to_id[self.mask_token]

    def tokenize(self, text):
        return [token for token in text.split()]

    def convert_tokens_to_string(self, tokens):
        return ' '.join(tokens)

    def convert_tokens_to_ids(self, tokens):
        return [self.token_to_id[token] for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.id_to_token[id] for id in ids]

# Step 3: Tokenize the text
def tokenize_text(text_data, tokenizer):
    tokenized_data = [tokenizer.tokenize(line) for line in text_data]
    return tokenized_data

# Step 4: Mask some of the tokens
def mask_tokens(tokenized_data, tokenizer, mask_probability=0.15):
    masked_data = []
    for line in tokenized_data:
        masked_line = []
        for token in line:
            if random.random() < mask_probability:
                masked_line.append(tokenizer.mask_token)
            else:
                masked_line.append(token)
        masked_data.append(masked_line)
    return masked_data

# Step 5: Predict the masked tokens using the model
def predict_masked_tokens(masked_data, tokenizer, model):
    predicted_data = []
    for masked_sentence in masked_data:
        masked_sentence_str = tokenizer.convert_tokens_to_string(masked_sentence)
        inputs = torch.tensor([tokenizer.convert_tokens_to_ids(masked_sentence)])
        with torch.no_grad():
            outputs = model(inputs)
        predictions = outputs.logits
        predicted_tokens = [tokenizer.convert_ids_to_tokens([token_id])[0] if token != tokenizer.mask_token else tokenizer.convert_ids_to_tokens([token_id])[0] for token_id, token in zip(torch.argmax(predictions, dim=-1)[0].tolist(), masked_sentence)]
        predicted_data.append(predicted_tokens)
    return predicted_data

if __name__ == "__main__":
    preprocess_text_dataset()
    CustomTokenizer()
    tokenize_text()
    mask_tokens()
    predict_masked_tokens()
