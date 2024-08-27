"""
Mask random words in the input text and finds the indices of the 
[MASK] text.
"""
import sys
import random
from typing import List, Tuple


def mask_words(clinical_notes_list: List[str],
               train_text: List[str],
               test_text: List[str],
               mask_prob: float, 
               mask_token: str) -> List[str]:
    """
    Mask words in the input text with a probability of 0.15.
    """
    # Reconfigure the standard output stream to use UTF-8 encoding.
    sys.stdout.reconfigure(encoding="utf-8")
    
    # Create empty lists
    masked_clinical_notes = []
    train_masked_notes = []
    test_masked_notes = []

    # Iterate over the the clinical notes
    for sentence in clinical_notes_list:
        # Split the input sentence into a list of words.
        words = sentence.split()
        # Create a new list to store the masked words.
        masked_sentence = []
        # Iterate over words
        for word in words:
            # Replace randomly the word  with the [MASK].
            if random.random() < mask_prob:
                # Append the [MASK] to the masked sentece
                masked_sentence.append(mask_token)
            else:
                # Append original word list
                masked_sentence.append(word)
        # Append the edited masked clinical notes to the list
        masked_clinical_notes.append(" ".join(masked_sentence))
    
    # Iterate over the the train clinical notes
    for sentence in train_text:
        # Split the input sentence into a list of words.
        words = sentence.split()
        # Create a new list to store the masked words.
        train_masked_sentence = []
        # Iterate over words
        for word in words:
            # Replace randomly the word  with the [MASK].
            if random.random() < mask_prob:
                # Append the [MASK] to the masked sentece
                train_masked_sentence.append(mask_token)
            else:
                # Append original word list
                train_masked_sentence.append(word)
        # Append the masked train notes to the list
        train_masked_notes.append(" ".join(train_masked_sentence))

    # Iterate over the the train clinical notes
    for sentence in test_text:
        # Split the input sentence into a list of words.
        words = sentence.split()
        # Create a new list to store the masked words.
        test_masked_sentence = []
        # Iterate over words
        for word in words:
            # Replace randomly the word  with the [MASK].
            if random.random() < mask_prob:
                # Append the [MASK] to the masked sentece
                test_masked_sentence.append(mask_token)
            else:
                # Append original word list
                test_masked_sentence.append(word)
        # Append the masked train notes to the list
        test_masked_notes.append(" ".join(test_masked_sentence))

    return masked_clinical_notes, train_masked_notes, test_masked_notes


def find_mask_indices(test_masked_notes)-> List[int]:
    """
    Find the indices of [MASK] words in the masked clinical notes
    """
    # Create an empty list
    mask_indices = []

    # Iterate over the masked clinical notes.
    for i, sentence in enumerate(test_masked_notes):
        # Create the starting point
        start = 0
        # Iterate ove the sentence
        while start < len(sentence):
            # Find the lowest index value of the first occurence of [MASK]
            index = sentence.find("[MASK]", start)
            # Create condition statement to break the loop if [MASK] is not found.
            if index == -1:
                break
            # Append the indices
            mask_indices.append([i, index])
            # Move the starting point to the next position after [MASK]
            start = index + len("[MASK]")
    
    return mask_indices


if __name__=="__main__":
    mask_words()
    find_mask_indices()
