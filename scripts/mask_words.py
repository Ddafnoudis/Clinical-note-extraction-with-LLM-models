"""
"""
import sys
import random
from typing import List


def mask_words(clinical_notes_list: List[str], mask_prob: float, mask_token: str) -> List[str]:
    """
    Mask words in the input text with a probability of 0.15.
    """
    # Reconfigure the standard output stream to use UTF-8 encoding.
    sys.stdout.reconfigure(encoding="utf-8")
    masked_clinical_notes = []
    
    for sentence in clinical_notes_list:
        words = sentence.split()
        masked_sentence = []
        
        for word in words:
            if random.random() < mask_prob:
                masked_sentence.append(mask_token)
            else:
                masked_sentence.append(word)
        
        masked_clinical_notes.append(" ".join(masked_sentence))
    
    return masked_clinical_notes


if __name__=="__main__":
    mask_words()
