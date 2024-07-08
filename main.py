"""
Testing Llama3 model for extraction of relevant words
"""
# Import libraries
import os
import pandas as pd
from scripts.llama_model import model, model_danish
from scripts.gen_dataframe import gen_dataframe

# Define the keyword
KEYWORD = input("Enter the word that you are interested in: ")


def main():
    # Condition statemet
    if os.path.exists("dataset_notes/clin_note_df.tsv") and os.path.exists("dataset_notes/clin_note_danish_df.tsv"):
        print("File exists")
    else:
        print("Generating a Dataframe with clinical notes")
        # Generate the DataFrame 
        gen_dataframe()

    # Parse the dataset
    dataframe = pd.read_csv("dataset_notes/clin_note_df.tsv", sep="\t", dtype=object)
    dataframe_danish = pd.read_csv("dataset_notes/clin_note_danish_df.tsv", sep="\t", dtype=object)
    # Response of the model (English)    
    response = model(df=dataframe, keyword=KEYWORD)
    print(response, "\n\n")
    # Respond of the model (Dansish)
    response_danish = model_danish(df_da=dataframe_danish, keyword=KEYWORD)
    print(response_danish)

if __name__ == "__main__":
    main()
