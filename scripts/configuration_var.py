"""
Define the configuartions file
"""
import yaml
from typing import Dict
from pathlib import Path


def parse_config_files(fname)-> Dict[str, Path]:
    """
    Parse the files
    """
    # Load the configuration file
    with open(fname) as stream:
        config = yaml.safe_load(stream)
    # Define the configuration as a dictionary
    config["DF_DK"] = str(config["DF_DK"])
    config["TOKENIZER"] = str(config["TOKENIZER"])
    config["LLM_MODEL"] = str(config["LLM_MODEL"])
    config["PARAMS"] = str(config["PARAMS"])
    config["MASK_TOKEN"] = str(config["MASK_TOKEN"])
    config["MASK_PROB"] = float(config["MASK_PROB"])
    config["MASK_TOKEN_ID"] = int(config["MASK_TOKEN_ID"])

    return config


if __name__=="__main__":
    parse_config_files()
