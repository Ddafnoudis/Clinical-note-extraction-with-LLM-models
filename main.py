"""
A pipeline for LLama-3.1-8B models training in Danish clinical text data
"""
from scripts.llm_pipeline import llm_pipeline
from scripts.configuration_var import parse_config_files


def main():
    # Parse files from configuration file
    config = parse_config_files(fname='configuration.yaml')	

    llm_pipeline(df_dk_path=config["DF_DK"], tokenizer_model=config["TOKENIZER"],
                 model=config["LLM_MODEL"], params_config=config["PARAMS"],
                 mask_token=config["MASK_TOKEN"], mask_prob=config["MASK_PROB"])
    

if __name__=="__main__":
    main()
