import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import tiktoken
from transformers import AutoTokenizer
import os

from src.config import ExperimentConfig

def get_tokenizer(config: ExperimentConfig):
    if config.data.tokenizer_type == 'tiktoken':
        tokenizer = tiktoken.get_encoding(config.data.tokenizer_name)
        pad_token_id = tokenizer.eot_token
        # Update vocab_size from tokenizer
        config.model.vocab_size = tokenizer.n_vocab
    elif config.data.tokenizer_type == 'huggingface':
        tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.pad_token_id
    else:
        raise ValueError(f"Unknown tokenizer type: {config.data.tokenizer_type}")
    return tokenizer, pad_token_id

def load_and_tokenize_data(config: ExperimentConfig, tokenizer, pad_token_id):
    def encode(text):
        if config.data.tokenizer_type == 'tiktoken':
            tokens = tokenizer.encode(text.numpy().decode('utf-8'))
            tokens = tokens[:config.model.maxlen]
            padding_needed = config.model.maxlen - len(tokens)
            return np.array(tokens + [pad_token_id] * padding_needed, dtype=np.int32)
        else: # huggingface
            return tokenizer(
                text.numpy().decode('utf-8'),
                truncation=True,
                padding="max_length",
                max_length=config.model.maxlen,
                return_tensors="np"
            )['input_ids'].squeeze(0).astype(np.int32)

    def tf_encode(text):
        return tf.py_function(encode, [text], tf.int32)

    if config.data.dataset_name:
        ds = tfds.load(config.data.dataset_name, split=config.data.split, as_supervised=False)
        ds = ds.map(lambda x: x['text']) # Extract text field
    elif config.data.data_path:
        ds = tf.data.TextLineDataset(config.data.data_path)
    else:
        raise ValueError("Either dataset_name or data_path must be provided in the config.")

    ds = ds.map(tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
    return ds

def get_dataset(config: ExperimentConfig):
    tokenizer, pad_token_id = get_tokenizer(config)
    
    train_ds = load_and_tokenize_data(config, tokenizer, pad_token_id)

    # Simple placeholder for validation, you might want a proper validation set
    val_ds = train_ds.take(1) 

    train_ds = (
        train_ds.shuffle(config.data.shuffle_buffer_size)
        .batch(config.data.batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        val_ds.batch(config.data.batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, tokenizer, pad_token_id