"""
Testing Llama3 model for extraction of relevant words
"""
# Import libraries
import os
import ollama
import pandas as pd
from scripts.gen_dataframe import gen_dataframe

# Define the keyword
KEYWORD = input("Enter the word that you are interested in: ")

# Pull model
try:
    ollama.pull("llama3")
    print("Model has been pulled")
except Exception as error:
    print("Error", error)


def main():
    """
    Test Llama3-8b for feature extraction
    """
    # Condition statemet
    if os.path.exists("dataset_notes/clin_note_df.tsv"):
        print("File exists")
    else:
        print("Generating a Dataframe with clinical notes")
        # Generate the DataFrame 
        gen_dataframe()

    # Parse the dataset
    df = pd.read_csv("dataset_notes/clin_note_df.tsv", sep="\t", dtype=object)
    # print(df.head());exit()

    # Create an empty list
    cli_notes_list = []
    # Define the clinical notes
    clinical_notes = df["clinical_notes"]
    # Append notes to a list
    for notes in clinical_notes:
        cli_notes_list.append(notes)
    
    # Define the message for the model
    message = [
    {"role": "system", "content": "You are an excellent Danish physician in extracting relevant words from clinical notes"},
    {"role": "user", "content": "Given the following clinical notes:\n\n".join(cli_notes_list) + "\n\nFind clinically relevant words from the clinical list that are related to the keyword: " + KEYWORD}
    ]
    
    # Response based on model and message
    try:
        response = ollama.chat(model='llama3', messages=message)
        return print(response['message']['content'])
    # Capture an error  
    except Exception as error:
        print("Error", error)
        return "Error"
    

if __name__ == "__main__":
    main()