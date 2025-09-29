#!/bin/bash

# 🎵 MusicGen + LoRA Training Script for 3-Hour Training on A100 GPU
# Optimized for A100 40GB with 48GB RAM
# Fixed version with audio format and precision fixes

echo "🎵 Starting MusicGen + LoRA Training - 3 Hour Target (Fixed Version)"
echo "===================================================================="

# Set environment variables for optimal A100 performance
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16

# Check GPU availability
echo "🔍 Checking GPU availability..."
nvidia-smi

# Check if training data exists
if [ ! -f "training_data/training_data_filtered.csv" ]; then
    echo "❌ Filtered training data not found! Please run data preparation first."
    echo "💡 Make sure you have run the data filtering script to create training_data_filtered.csv"
    exit 1
fi

echo "📊 Training data found: $(wc -l < training_data/training_data_filtered.csv) samples"

# Create output directory
mkdir -p checkpoints
mkdir -p generated_samples

# Run training with 3-hour optimized settings for A100 40GB
echo "🚀 Starting 3-hour training with fixes..."
echo "🔧 Fixes applied:"
echo "   - Mono audio conversion (8-channel → 1-channel)"
echo "   - Float32 precision (no mixed precision issues)"
echo "   - Missing file filtering"
echo "   - Enhanced error handling"
echo ""

python musicgen_lora_main.py \
    --metadata_file "training_data/training_data_filtered.csv" \
    --target_categories piano flute guitar meditation relaxation \
    --lora_rank 24 \
    --lora_alpha 48.0 \
    --lora_dropout 0.1 \
    --learning_rate 1.5e-4 \
    --batch_size 8 \
    --num_epochs 10 \
    --device cuda \
    --test_generation

echo ""
echo "🎉 3-hour training completed!"
echo "📁 Checkpoints saved in: checkpoints/"
echo "🎵 Test samples generated in: generated_samples/"
echo ""
echo "📊 Expected results:"
echo "   - No more '8 channels' errors"
echo "   - No more precision mismatch errors"
echo "   - Proper training loss values"
echo "   - Successful test generation"