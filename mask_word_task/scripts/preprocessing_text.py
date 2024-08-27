"""
Preprocess the clinical text data LLM training
"""
import sys
import pandas as pd
from typing import List
from pathlib import Path
from sklearn.model_selection import train_test_split


def preprocessing_text(df_danish_path: Path)-> List[str]:
    """
    Preprocessing the danish clinical text data
    """
    # Reconfigure the standard output stream to use UTF-8 encoding.
    sys.stdout.reconfigure(encoding="utf-8")
    # Parse dataset
    df_dk = pd.read_csv(df_danish_path, sep="\t", dtype=object)
    # Preproces and extract clinical notes
    clinical_notes_list = df_dk["clinical_notes"].tolist()
    # Split clinical notes into train and test sets
    train_text, test_text = train_test_split(clinical_notes_list, train_size=0.2, random_state=42) 
    # print(type(train_text))

    return train_text, test_text, clinical_notes_list


if __name__=="__main__":
    preprocessing_text()
