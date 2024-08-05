"""
"""
import sys
import torch
import random
from typing import List


def mask_words(test_text: List[str], mask_prob: float, mask_token: str) -> List[str]:
    """
    Mask words in the input text with a probability of 0.15.
    """
    # Reconfigure the standard output stream to use UTF-8 encoding.
    sys.stdout.reconfigure(encoding="utf-8")
    masked_test_set = []
    
    for sentence in test_text:
        words = sentence.split()
        masked_sentence = []
        
        for word in words:
            if random.random() < mask_prob:
                masked_sentence.append(mask_token)
            else:
                masked_sentence.append(word)
        
        masked_test_set.append(" ".join(masked_sentence))
    
    return masked_test_set


if __name__=="__main__":
    mask_words()
