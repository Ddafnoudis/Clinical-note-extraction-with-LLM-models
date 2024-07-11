"""
Condition statements based on presence of parameters
Parametes:  1) Clinical notes datasets (English and Danish)
            2) Folder with those datasets
"""
import os
from pathlib import Path
from pandas import DataFrame
from scripts.gen_dataframe import gen_dataframe


def parse_data(data_folder_path: Path, df_eng_path: Path, df_dk_path: Path)-> DataFrame:
    """
    Condition statements
    """
    # Condition statemet
    if os.path.exists(df_eng_path) and os.path.exists(df_dk_path):
        print("Folder and files exist")
    elif os.path.exists(data_folder_path):
        print("Folder exists! Generating a Dataframe with clinical notes")
        # Generate the DataFrame 
        gen_dataframe()
    else:
        os.mkdir(data_folder_path)
        print("Generating a Dataframe with clinical notes")
        # Generate the DataFrame 
        gen_dataframe()


if __name__ == "__main__":
    parse_data()
