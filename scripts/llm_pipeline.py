"""
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from scripts.define_model import model_config
from scripts.tokenize_input import tokenize_input
from scripts.query_tensor import query_tensor
from scripts.key_tensor import key_tensor
from scripts.embendding_layer import embedding_layer
from scripts.self_attention import self_attention_qkv
from scripts.param_config import param_configuration
from scripts.final import LLama3_1
from scripts.preprocessing_text import preprocessing_text
from scripts.tokenizer_preprocess import preprocess_tokenizer
from scripts.multi_head_attention import multi_head_attention
from scripts.swiglu_act_funtion import swiglu_activation_function
from scripts.predict import predict_masked_tokens

from scripts.mask_words import mask_words, find_mask_indices


def llm_pipeline(df_dk_path: Path, tokenizer_model: Path,
                 model: Path, params_config: Path,
                 mask_token: str, mask_prob: float,
                 mask_token_id=int):
    """
    """
    # Configure the model
    tokenizer_model, model, params_config = model_config(tokenizer=tokenizer_model, model=model, params=params_config)
    # print(f"\nModel {model}");exit()
    # Define the parameterss
    dim, n_layers, n_heads, n_kv_heads, vocab_size, multiple_of, ffn_dim_multiplier, norm_eps, rope_theta = param_configuration(config=params_config)
    # Define the tokenizer
    tokenizer = preprocess_tokenizer(tokenizer_model=tokenizer_model)
    # Preprocess the clinical text data
    train_text, test_text, clinical_notes_list= preprocessing_text(df_danish_path=df_dk_path)
    
    masked_clinical_notes, train_masked_notes, test_masked_notes = mask_words(clinical_notes_list=clinical_notes_list, 
                                       train_text=train_text,
                                       test_text=test_text,
                                       mask_token=mask_token, 
                                       mask_prob=mask_prob)
    
    # print(masked_clinical_notes)
    mask_indices = find_mask_indices(masked_clinical_notes=masked_clinical_notes)
    # print(mask_indices);exit()
    # Tokenize words
    tokens = tokenize_input(tokenizer=tokenizer, masked_clinical_notes=masked_clinical_notes)
    print(f"Tokens tensor shape: {tokens.shape}")
    # Define the embedding tokens and normalize them
    token_embeddings, token_embeddings_unnormal = embedding_layer(model=model, vocab_size=vocab_size, dim=dim, tokens=tokens, norm_eps=norm_eps)
    print(f"\nToken Embeddings Shape: {token_embeddings.shape}\n")
    print(f"\nToken Embeddings Unnormalized shape: {token_embeddings_unnormal.shape}\n")
    q_layer_0, q_per_token_rotated, freqs_cis = query_tensor(model=model, n_heads=n_heads, dim=dim, token=tokens, token_embeddings=token_embeddings, rope_theta=rope_theta)

    model = LLama3_1(model=model, n_layers=n_layers, n_heads=n_heads,
                     dim=dim, n_kv_heads=n_kv_heads, norm_eps=norm_eps,
                     freqs_cis=freqs_cis)
    # Check the number of trainable parameters
    # print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


    num_epochs=10 
    learning_rate=1e-4
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i, masked_note in enumerate(train_masked_notes):
            optimizer.zero_grad()

            # Tokenize the masked note
            tokens = tokenize_input(tokenizer=tokenizer, masked_clinical_notes=[masked_note])
            token_embeddings, _ = embedding_layer(model=model, vocab_size=vocab_size, dim=dim, tokens=tokens, norm_eps=norm_eps)

            # Forward pass
            outputs = model(token_embeddings)

            # Compute loss
            loss = criterion(outputs.view(-1, vocab_size), tokens.view(-1))
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_masked_notes)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("Training complete.")

    # predictions = predict_masked_tokens(mask_indices=mask_indices, 
    #                                     model=model, tokenizer=tokenizer, 
    #                                     masked_clinical_notes=masked_clinical_notes)

if __name__ =="__main__":
    llm_pipeline()
