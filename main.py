"""
Testing Llama3 model for extraction of relevant words
"""
# Import libraries and modules
import pandas as pd
from pathlib import Path
from scripts.parse_data import parse_data
from scripts.llama_test import llama_test
from scripts.llama_model import model, model_danish
from scripts.llama_train_in_danish import train_llama3

# Define keyword and paths
# KEYWORD = input("Enter the word that you are interested in: ")
MY_TOKEN = "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" #input("Provide your token: ")
LLM_MODEL = "LLama3_instruct/Meta-Llama-3-8B-Instruct"
DATA_FOLDER = Path("~/Desktop/Clinical-note-extraction-with-LLM-models/dataset_notes").expanduser()
DF_ENG = Path("~/Desktop/Clinical-note-extraction-with-LLM-models/dataset_notes/clin_note_df.tsv").expanduser()
DF_DK = Path("~/Desktop/Clinical-note-extraction-with-LLM-models/dataset_notes/clin_note_danish_df.tsv").expanduser()

# Set the batch size and number of epochs for training
BATCH_SIZE = 16
NUM_EPOCHS = 5

def main():
    # Condition statement
    parse_data(data_folder_path=DATA_FOLDER, df_eng_path=DF_ENG, df_dk_path=DF_DK)
    # Parse dataset
    df_eng = pd.read_csv(DF_ENG, sep="\t", dtype=object)
    df_dk = pd.read_csv(DF_DK, sep="\t", dtype=object)


    # Train the model
    llama_test(df_dk=df_dk, llm_model=LLM_MODEL, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS)

    # Train the Llama3 model 
    attempt_response = train_llama3(df_dk=df_dk)
    print(attempt_response);exit()

    # Response of the model (English)    
    response = model(df_eng=df_eng, keyword=KEYWORD)
    print(response, "\n\n")
    # Respond of the model (Dansish)
    response_danish = model_danish(df_dk=df_dk, keyword=KEYWORD)
    print(response_danish)


if __name__ == "__main__":
    main()
