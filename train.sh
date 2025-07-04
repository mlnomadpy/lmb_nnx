#!/bin/bash
set -ex

# Install dependencies
pip install -r requirements.txt

# Create a directory for the output
WORKDIR="output"
mkdir -p $WORKDIR

# Force TensorFlow to only use CPU, preventing conflicts with JAX on the TPU.
export CUDA_VISIBLE_DEVICES=""

# Run training
python main.py \
  --workdir=$WORKDIR \
  --config.dataset_name='wikitext' \
  --config.dataset_config_name='wikitext-2-raw-v1' \
  --config.tokenizer_name='gpt2' \
  --config.per_device_batch_size=2 \
  --config.num_train_steps=1000 \
  --config.eval_every_steps=100 \
  --config.text_column_name='text' 