"""
Testing Llama3 model for extraction of relevant words
"""
# Import libraries and modules
import pandas as pd
from pathlib import Path
from scripts.parse_data import parse_data
from scripts.llama_model import model_danish 
from sklearn.model_selection import train_test_split

# Define keyword and paths
KEYWORD = input("Enter the word that you are interested in: ")

DATA_FOLDER = Path("~/Desktop/Clinical-note-extraction-with-LLM-models/dataset_notes").expanduser()
DF_ENG = Path("~/Desktop/Clinical-note-extraction-with-LLM-models/dataset_notes/clin_note_df.tsv").expanduser()
DF_DK = Path("~/Desktop/Clinical-note-extraction-with-LLM-models/dataset_notes/clin_note_danish_df.tsv").expanduser()
RESULTS = Path("~/Desktop/Clinical-note-extraction-with-LLM-models/results").expanduser()
CHAT_OUTPUT = Path("~/Desktop/Clinical-note-extraction-with-LLM-models/results/response_list.txt").expanduser()


# Set the batch size and number of epochs for training
BATCH_SIZE = 5


def main():
    # Condition statement
    parse_data(data_folder_path=DATA_FOLDER, df_eng_path=DF_ENG, df_dk_path=DF_DK, results=RESULTS)
    # Parse dataset
    df_eng = pd.read_csv(DF_ENG, sep="\t", dtype=object)
    df_dk = pd.read_csv(DF_DK, sep="\t", dtype=object)

    # Preproces and extract clinical notes
    clinical_notes_list = df_dk["clinical_notes"].tolist()
    # Split clinical notes into train and test sets
    train_text, test_text = train_test_split(clinical_notes_list, train_size=0.2, random_state=42) 
    
    # Respond of the model (Dansish)
    response_danish = model_danish(df_dk=df_dk, keyword=KEYWORD, batch_size=BATCH_SIZE, chat_output=CHAT_OUTPUT)
    print(response_danish)


if __name__ == "__main__":
    main()
