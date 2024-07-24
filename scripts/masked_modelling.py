"""
Train Llama3-8B into masked language model tasks.
Perform a performance metric calulation
"""
import torch
import random
import transformers
import numpy as np
from pathlib import Path
from typing import List, Any, Dict
from transformers import pipeline, AutoTokenizer, LlamaForCausalLM


def mask_words(train_text: List[str], test_text: List[str], llm_model: str, model_path: Path) -> Any:
   
    pipe = pipeline("fill-mask", model=llm_model)
    tokenizer = AutoTokenizer.from_pretrained(llm_model)
    model = LlamaForCausalLM.from_pretrained(llm_model)

    masked_token = "[MASK]"
    masked_probability = 0.20


    masked_text = []
    for sentence in train_text:
        words = sentence.split()
        masked_sen = [
            masked_token if random.random() < masked_probability else word for word in words
        ]
        masked_text.append(" ".join(masked_sen))
    
    print(masked_text);exit()
    
        

if __name__ == "__main__":
    mask_words()
