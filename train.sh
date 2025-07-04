#!/bin/bash
set -ex

# Install dependencies
pip install -r requirements.txt

# Create a directory for the output
WORKDIR="output"
mkdir -p $WORKDIR

# Run training
python main.py \
  --workdir=$WORKDIR \
  --config.hf_dataset='wikitext' \
  --config.hf_dataset_config_name='wikitext-2-raw-v1' \
  --config.hf_tokenizer='gpt2' \
  --config.per_device_batch_size=2 \
  --config.num_train_steps=1000 \
  --config.eval_every_steps=100 \
  --config.text_column_name='text' 