"""
Train llama3 on in Danish dataset
"""
import ollama
import tensorflow
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, wandb
from trl import SFTTrainer, setup_chat_format

from huggingface_hub import login


def train_llama3(token_id: str, llm_model: str):
    """
    """
    login(token = token_id)

    wandb.login(key=token_id)
    run = wandb.init(
        project='Fine-tune Llama 3 8B on Medical Dataset', 
        job_type="training", 
        anonymous="allow"
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_implementation
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model, tokenizer = setup_chat_format(model, tokenizer)
    pipe = pipeline(
        "text-generation",
        model=llm_model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
    )

    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]

    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipe(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    assistant_response = outputs[0]["generated_text"][-1]["content"]
    print(assistant_response)
    

if __name__ == "__main__":
    train_llama3()