"""
Testing Llama3 model for extraction of relevant words
"""
# Import libraries
import os
import pandas as pd
from scripts.llama_model import model
from scripts.gen_dataframe import gen_dataframe

# Define the keyword
KEYWORD = input("Enter the word that you are interested in: ")


def main():
    # Condition statemet
    if os.path.exists("dataset_notes/clin_note_df.tsv"):
        print("File exists")
    else:
        print("Generating a Dataframe with clinical notes")
        # Generate the DataFrame 
        gen_dataframe()

    # Parse the dataset
    dataframe = pd.read_csv("dataset_notes/clin_note_df.tsv", sep="\t", dtype=object)

    # Response of the model    
    response = model(df=dataframe, keyword=KEYWORD)
    print(response)


if __name__ == "__main__":
    main()
