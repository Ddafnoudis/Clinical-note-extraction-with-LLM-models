"""
Interact with the architecture of the Llama-3.1 model for masking words prediction tasks.
Load the model, tokenizer and the parameters. The clinical text is in Danish. 
"""
import torch
from pathlib import Path
from scripts.key_tensor import key_tensor
from scripts.query_tensor import query_tensor
from scripts.define_model import model_config
from scripts.final import final_all_layers_merged
from scripts.tokenize_input import tokenize_input
from scripts.embendding_layer import embedding_layer
from scripts.param_config import param_configuration
from scripts.self_attention import self_attention_qkv
from scripts.preprocessing_text import preprocessing_text
from scripts.mask_words import mask_words, find_mask_indices
from scripts.tokenizer_preprocess import preprocess_tokenizer
from scripts.multi_head_attention import multi_head_attention
from scripts.swiglu_act_funtion import swiglu_activation_function


def llm_pipeline(df_dk_path: Path, tokenizer_model: Path,
                 model: Path, params_config: Path,
                 mask_token: str, mask_prob: float,
                 mask_token_id=int):
    """
    Params: 1) Load the model, tokenizer and the parameters
            2) Define the tokenizer
            3) Preprocess the clinical text data. Mask the words and tokenize
            4) Create the embedding layer and implement Sefl Attention
            5) Implement Multi-Head Attention
            6) Generate the final embedding layer
    """
    # Configure the model
    tokenizer_model, model, params_config = model_config(tokenizer=tokenizer_model, model=model, params=params_config)
    # Define the parameterss
    dim, n_layers, n_heads, n_kv_heads, vocab_size, multiple_of, ffn_dim_multiplier, norm_eps, rope_theta = param_configuration(config=params_config)
    # Define the tokenizer
    tokenizer = preprocess_tokenizer(tokenizer_model=tokenizer_model)
    # Preprocess the clinical text data
    train_text, test_text, clinical_notes_list= preprocessing_text(df_danish_path=df_dk_path)
    # Mask the words from the train, test clinical notes
    masked_clinical_notes = mask_words(clinical_notes_list=clinical_notes_list, 
                                       train_text=train_text,
                                       test_text=test_text,
                                       mask_token=mask_token, 
                                       mask_prob=mask_prob)
    
    masked_clinical_notes = masked_clinical_notes[1]
    # print(masked_clinical_notes);exit()
    # Find the indices of the masked tokens
    test_mask_indices = find_mask_indices(test_masked_notes=masked_clinical_notes)
    # Tokenize words
    tokens = tokenize_input(tokenizer=tokenizer, masked_clinical_notes=masked_clinical_notes)
    # Define the embedding tokens and normalize them
    token_embeddings, token_embeddings_unnormal = embedding_layer(model=model, vocab_size=vocab_size, dim=dim, tokens=tokens, norm_eps=norm_eps)
    # Calculate the first layer of the query tensor 
    q_layer_0, q_per_token_rotated, freqs_cis = query_tensor(model=model, n_heads=n_heads, dim=dim, token=tokens, token_embeddings=token_embeddings, rope_theta=rope_theta)
    # Calculate the key tensor of the first layer
    k_layer_0, k_per_token_rotated = key_tensor(freqs_cis=freqs_cis, 
                                                model=model, 
                                                n_kv_heads=n_kv_heads, 
                                                dim=dim, 
                                                token_embeddings=token_embeddings)
    
    # Implement Self-Attention
    v_layer_0 = self_attention_qkv(model=model,n_kv_heads=n_kv_heads,
                                   token_embeddings=token_embeddings, dim=dim,
                                   tokens=tokens, q_per_token_rotated=q_per_token_rotated,
                                   k_per_token_rotated=k_per_token_rotated)
    # Implement Multi-Head Attention     
    embedding_after_edit_normalized = multi_head_attention(n_heads=n_heads, q_layer_0=q_layer_0, 
                                                           k_layer_0=k_layer_0, v_layer_0=v_layer_0,
                                                           tokens=tokens, token_embeddings=token_embeddings,
                                                           freqs_cis=freqs_cis, norm_eps=norm_eps, model=model,
                                                           token_embeddings_unnormal=token_embeddings_unnormal)
    
    # Activate SwiGLU Activation Function to adjust the importance of each word or phrase
    output_after_feedforward = swiglu_activation_function(embedding_after_edit_normalized=embedding_after_edit_normalized,
                                                          model=model)
    # Merge all the layers to obtain the final embedding
    final_embedding = final_all_layers_merged(token_embeddings_unnormal=token_embeddings_unnormal, 
                                              model=model, n_layers=n_layers, n_heads=n_heads,
                                              dim=dim, n_kv_heads=n_kv_heads, norm_eps=norm_eps,
                                              freqs_cis=freqs_cis)
    # Create a loop to iterate over the mask tokens and replace them with the predicted next token
    # Find the positions of [MASK] tokens
    mask_positions = [i for i, token in enumerate(tokens) if token == '[MASK]']
    	
    for mask_pos in mask_positions:
        # Calculate logits for the current [MASK] position
        logits = torch.matmul(final_embedding[mask_pos], model["output.weight"].T)

        # Find the index of the maximum value to determine the next token
        next_token = torch.argmax(logits, dim=-1)

        # You can now use next_token to replace the [MASK] at this position
        # For example, you could update the tokens list:
        tokens[mask_pos] = next_token.item()

    print(f"Replaced [MASK] at position {mask_pos} with token {next_token.item()}")

    # After the loop, you can convert tokens back to text if needed
    


if __name__ =="__main__":
    llm_pipeline()
