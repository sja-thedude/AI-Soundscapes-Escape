# 🎶 AI Soundscapes

AI Soundscapes is a research & development project that explores generating immersive, dynamic sound environments using AI models.
It leverages MusicGen with LoRA fine-tuning to create realistic, controllable audio experiences for games, meditation, film, and interactive applications.

## ✨ What it Does
	•	Generates custom background soundscapes (e.g., nature, sci-fi, city life, ambient music).
	•	Supports fine-tuned styles via LoRA (Low-Rank Adaptation) for efficient training.
	•	Produces high-quality audio clips from simple text prompts.
	•	Runs reproducibly across environments with pinned dependencies.

## 🔧 Key Features
	•	LoRA Training Support → Optimizer only updates LoRA layers, keeping base model frozen.
	•	Stable Environment → Requirements pinned for reproducible results.
	•	Efficient GPU Usage → Reduced memory footprint, stable training on A100.
	•	Checkpoint & Gradient Tracking → LoRA params registered properly for saving and resuming.
	•	Torch Precision Control → Uses torch.float32 for stability, avoiding mixed precision issues.

## 🚀 Getting Started

Installation

git clone https://github.com/yourusername/ai-soundscapes.git
cd ai-soundscapes
pip install -r requirements_training.txt

Training with LoRA

python train.py --config configs/lora_config.json

Generate a Soundscape

python generate.py --prompt "gentle rainforest with birds and flowing water"


## 📦 Dependencies
	•	PyTorch (stable release)
	•	Transformers
	•	Librosa
	•	MusicGen + LoRA support

All versions are pinned in requirements_training.txt for reproducibility.


## 🌍 Use Cases
	•	🎮 Game Development → Dynamic background audio that adapts to gameplay.
	•	🧘 Meditation & Wellness → Relaxing natural soundscapes.
	•	🎬 Film & Animation → Quick prototyping of ambient audio.
	•	🤖 AI Research → Experimenting with LoRA fine-tuning for music/audio generation.

## Current Status
	•	Stable LoRA training for up to 3-hour sessions on A100 GPUs.
	•	Produces consistent audio output with reproducible results.
	•	Ready for extension into real-time or interactive applications.
