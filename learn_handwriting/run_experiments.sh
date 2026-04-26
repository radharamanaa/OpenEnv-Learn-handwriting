#!/bin/bash

# Experiment Runner for Handwriting SFT
# This script trains multiple LoRA versions with varying Rank and Alpha
# to find the best balance for the handwriting stroke policy.

# --- Configuration ---
# Load username from .env if possible
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

DATASET="datagen_sft/qwen25_finetune_data.jsonl"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
HUB_MODEL_ID="${HF_USERNAME:-abhijeetmishra101}/Qwen2.5-Letter-Drawing-Instructed"
EPOCHS=3

# Ensure we are in the repo root
export PYTHONPATH=$PYTHONPATH:.

echo "🚀 Starting Handwriting Fine-Tuning Experiments..."

# Define experiments: "rank alpha"
experiments=(
    "8 16"
    "16 16"
    "16 32"
    "16 64"
)

for exp in "${experiments[@]}"; do
    # Split the experiment string into r and a
    read -r r a <<< "$exp"
    
    REVISION="rev-r${r}-a${a}"
    OUTPUT_DIR="outputs/${REVISION}"
    
    echo "----------------------------------------------------------------"
    echo "🧪 RUNNING: Rank=$r, Alpha=$a --> Revision: $REVISION"
    echo "----------------------------------------------------------------"

    python datagen_sft/train_qwen_lora.py \
        --dataset "$DATASET" \
        --model_name "$MODEL_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --num_train_epochs "$EPOCHS" \
        --lora_r "$r" \
        --lora_alpha "$a" \
        --hub_model_id "$HUB_MODEL_ID" \
        --hub_revision "$REVISION" \
        --hub_private

    if [ $? -eq 0 ]; then
        echo "✅ Finished $REVISION and pushed to Hugging Face."
    else
        echo "❌ Experiment $REVISION failed."
        exit 1
    fi
done

echo "🎉 All experiments completed successfully!"
