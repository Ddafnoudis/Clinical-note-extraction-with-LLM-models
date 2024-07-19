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
LLM_MODEL = Path("~/Desktop/Clinical-note-extraction-with-LLM-models/AI-Sweden-Models/Llama-3-8B").expanduser()
MODEL_PATH = f"{LLM_MODEL}/model-00004-of-00004.safetensors"
TOKENIZER_PATH = Path("~/Desktop/Clinical-note-extraction-with-LLM-models/AI-Sweden-Models/Llama-3-8B/tokenizer.json").expanduser()
# Fine-tuned model name
NEW_MODEL=  Path("~/Desktop/Clinical-note-extraction-with-LLM-models/AI-Sweden-Models/Llama-3-8B/llama-3-8b-danish")
DATA_FOLDER = Path("~/Desktop/Clinical-note-extraction-with-LLM-models/dataset_notes").expanduser()
DF_ENG = Path("~/Desktop/Clinical-note-extraction-with-LLM-models/dataset_notes/clin_note_df.tsv").expanduser()
DF_DK = Path("~/Desktop/Clinical-note-extraction-with-LLM-models/dataset_notes/clin_note_danish_df.tsv").expanduser()

# Set the batch size and number of epochs for training
BATCH_SIZE = 1
NUM_EPOCHS = 1

def main():
    # Condition statement
    parse_data(data_folder_path=DATA_FOLDER, df_eng_path=DF_ENG, df_dk_path=DF_DK)
    # Parse dataset
    df_eng = pd.read_csv(DF_ENG, sep="\t", dtype=object)
    df_dk = pd.read_csv(DF_DK, sep="\t", dtype=object)

    
    # Train the model
    llama_test(df_dk=df_dk, llm_model=LLM_MODEL, model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH, batch_size=BATCH_SIZE, new_model=NEW_MODEL, num_epochs=NUM_EPOCHS)

    # Train the Llama3 model 
    # attempt_response = train_llama3(df_dk=df_dk)
    # print(attempt_response);exit()

    # Response of the model (English)    
    # response = model(df_eng=df_eng, keyword=KEYWORD)
    # print(response, "\n\n")
    # # Respond of the model (Dansish)
    # response_danish = model_danish(df_dk=df_dk, keyword=KEYWORD)
    # print(response_danish)


if __name__ == "__main__":
    main()
