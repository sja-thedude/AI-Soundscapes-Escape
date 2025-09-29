#!/usr/bin/env python3
"""
🎵 MusicGen + LoRA Training Script - Main Entry Point 🎵

Run this script on your A100 GPU supercomputer for training.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple
import json
import argparse
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Import our custom classes
from musicgen_lora_trainer import InstrumentDataset, LoRALayer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MusicGenLoRATrainer:
    """MusicGen + LoRA trainer for instrument soundscapes."""
    
    def __init__(self, 
                 model_name: str = "facebook/musicgen-small",
                 target_categories: List[str] = None,
                 lora_rank: int = 16,
                 lora_alpha: float = 32.0,
                 lora_dropout: float = 0.1,
                 learning_rate: float = 1e-4,
                 batch_size: int = 4,
                 num_epochs: int = 15,
                 device: str = "cuda"):
        
        self.model_name = model_name
        self.target_categories = target_categories or ['piano', 'flute', 'guitar', 'meditation', 'relaxation']
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        
        # Initialize model and LoRA
        self.model = None
        self.processor = None
        self.lora_layers = {}
        self.optimizer = None
        self.scheduler = None
        
        # Setup
        self._setup_model()
        self._setup_optimizer()
    
    def _setup_model(self):
        """Setup MusicGen model with LoRA."""
        try:
            from transformers import MusicgenForConditionalGeneration, AutoProcessor
            
            logger.info(f"🔄 Loading MusicGen model: {self.model_name}")
            
            # Load model and processor
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for stability
                device_map="auto"
            )
            
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Apply LoRA to target modules
            self._apply_lora()
            
            logger.info("✅ MusicGen model loaded with LoRA!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load MusicGen model: {e}")
            raise
    
    def _apply_lora(self):
        """Apply LoRA to target modules."""
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        logger.info(f"🔧 Applying LoRA to modules: {target_modules}")
        
        for name, module in self.model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    # Create LoRA layer
                    lora_layer = LoRALayer(
                        module.in_features,
                        module.out_features,
                        rank=self.lora_rank,
                        alpha=self.lora_alpha,
                        dropout=self.lora_dropout
                    ).to(self.device)
                    
                    # Store original forward method
                    original_forward = module.forward
                    
                    # Create new forward method with LoRA
                    def lora_forward(x):
                        return original_forward(x) + lora_layer(x)
                    
                    # Replace forward method
                    module.forward = lora_forward
                    
                    # Store LoRA layer
                    self.lora_layers[name] = lora_layer
                    
                    logger.info(f"✅ Applied LoRA to: {name}")
        
        logger.info(f"🎯 Applied LoRA to {len(self.lora_layers)} modules")
    
    def _setup_optimizer(self):
        """Setup optimizer for LoRA parameters only."""
        # Get only LoRA parameters
        lora_params = []
        for name, param in self.model.named_parameters():
            if 'lora' in name:
                lora_params.append(param)
                logger.info(f"Found LoRA parameter: {name}")
        
        if not lora_params:
            logger.warning("No LoRA parameters found! Using all parameters.")
            lora_params = list(self.model.parameters())
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            lora_params,
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
            eta_min=self.learning_rate * 0.1
        )
        
        logger.info(f"🔧 Optimizer setup with {len(lora_params)} LoRA parameters")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch with enhanced progress tracking."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Enhanced progress bar with more details
        progress_bar = tqdm(dataloader, 
                          desc=f"Epoch {epoch+1}/{self.num_epochs}",
                          unit="batch",
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move data to device
                audio = batch['audio'].to(self.device)
                prompts = batch['prompt']
                
                # Prepare inputs
                inputs = self.processor(
                    text=prompts,
                    audio=audio.cpu().numpy(),
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                # Generate audio for loss calculation
                with torch.cuda.amp.autocast(enabled=False):  # Disable mixed precision for stability
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=1.0
                )
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Enhanced progress bar updates
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/num_batches:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
            except Exception as e:
                logger.error(f"❌ Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"train_loss": avg_loss}
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader = None):
        """Main training loop with enhanced progress tracking."""
        logger.info("🚀 Starting MusicGen + LoRA training...")
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        # Overall training progress bar
        overall_progress = tqdm(range(self.num_epochs), desc="Training Progress", unit="epoch")
        
        for epoch in overall_progress:
            logger.info(f"📅 Epoch {epoch+1}/{self.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_dataloader, epoch)
            train_losses.append(train_metrics["train_loss"])
            
            # Save checkpoint every few epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"musicgen_instruments_lora_epoch_{epoch+1}.pth")
            
            # Update scheduler
            self.scheduler.step()
            
            # Update overall progress bar
            overall_progress.set_postfix({
                'train_loss': f'{train_metrics["train_loss"]:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            logger.info(f"📊 Epoch {epoch+1} - Train Loss: {train_metrics['train_loss']:.4f}")
        
        # Save final model
        self.save_checkpoint("musicgen_instruments_lora_final.pth")
        
        logger.info("🎉 Training completed!")
        return train_losses, val_losses
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'lora_layers': {name: layer.state_dict() for name, layer in self.lora_layers.items()},
            'config': {
                'model_name': self.model_name,
                'target_categories': self.target_categories,
                'lora_rank': self.lora_rank,
                'lora_alpha': self.lora_alpha,
                'lora_dropout': self.lora_dropout,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs
            }
        }
        
        torch.save(checkpoint, filename)
        logger.info(f"💾 Saved checkpoint: {filename}")
    
    def generate_test_samples(self, output_dir: str = "generated_samples"):
        """Generate test samples with instrument prompts - Fixed version."""
        logger.info("🎵 Generating test samples...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        test_prompts = [
            "Soothing flute meditation soundscape",
            "Calm piano and gentle guitar for relaxation",
            "Peaceful piano melody for mindfulness",
            "Gentle flute music for deep breathing",
            "Acoustic guitar for peaceful meditation",
            "Tranquil piano soundscape for healing",
            "Meditation music with soft flute and piano",
            "Relaxing guitar and piano for stress relief"
        ]
        
        self.model.eval()
        
        # Ensure model is in float32 for generation
        self.model = self.model.float()
        
        with torch.no_grad():
            for i, prompt in enumerate(test_prompts):
                try:
                    logger.info(f"🎼 Generating: {prompt}")
                    
                    inputs = self.processor(
                        text=[prompt],
                        padding=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Generate audio with consistent precision
                    audio_values = self.model.generate(
                        **inputs,
                        max_new_tokens=int(30 * 10),  # 30 seconds
                        do_sample=True,
                        temperature=0.8,
                        top_k=250,
                        top_p=0.0
                    )
                    
                    # Save audio
                    audio = audio_values[0, 0].cpu().numpy()
                    
                    # Normalize
                    if np.max(np.abs(audio)) > 0:
                        audio = audio / np.max(np.abs(audio)) * 0.8
                    
                    filename = f"{output_dir}/test_sample_{i+1:02d}_{prompt.replace(' ', '_')[:30]}.wav"
                    sf.write(filename, audio, 32000)
                    
                    logger.info(f"✅ Saved: {filename}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to generate {prompt}: {e}")
        
        logger.info(f"🎉 Generated {len(test_prompts)} test samples in {output_dir}/")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="MusicGen + LoRA Training for Instruments")
    parser.add_argument("--metadata_file", type=str, default="training_data/training_data_filtered.csv",
                       help="Path to training metadata CSV")
    parser.add_argument("--target_categories", nargs="+", 
                       default=["piano", "flute", "guitar", "meditation", "relaxation"],
                       help="Target categories for training")
    parser.add_argument("--lora_rank", type=int, default=24, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=48.0, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--learning_rate", type=float, default=1.5e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--test_generation", action="store_true", help="Generate test samples after training")
    
    args = parser.parse_args()
    
    print("🎵 MusicGen + LoRA Training for Instrument Categories")
    print("=" * 60)
    print(f"🎯 Target categories: {args.target_categories}")
    print(f"🔧 LoRA config: rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"📊 Training config: lr={args.learning_rate}, batch_size={args.batch_size}, epochs={args.num_epochs}")
    print(f"🖥️  Device: {args.device}")
    
    # Check if metadata file exists
    if not os.path.exists(args.metadata_file):
        logger.error(f"❌ Metadata file not found: {args.metadata_file}")
        return
    
    # Create dataset
    logger.info("📊 Creating dataset...")
    dataset = InstrumentDataset(
        metadata_file=args.metadata_file,
        target_categories=args.target_categories
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders (optimized for 16-core CPU)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,  # Optimized for 16-core CPU
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,  # Optimized for 16-core CPU
        pin_memory=True
    )
    
    logger.info(f"📊 Training samples: {len(train_dataset)}")
    logger.info(f"📊 Validation samples: {len(val_dataset)}")
    
    # Create trainer
    trainer = MusicGenLoRATrainer(
        target_categories=args.target_categories,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=args.device
    )
    
    # Train
    try:
        train_losses, val_losses = trainer.train(train_dataloader, val_dataloader)
        
        # Generate test samples
        if args.test_generation:
            trainer.generate_test_samples()
        
        logger.info("🎉 Training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()