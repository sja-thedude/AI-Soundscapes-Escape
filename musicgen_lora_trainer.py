#!/usr/bin/env python3
"""
🎵 MusicGen + LoRA Training Script - Core Classes 🎵

This script contains the core training classes for MusicGen + LoRA fine-tuning.
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InstrumentDataset(Dataset):
    """Dataset for instrument soundscape training."""
    
    def __init__(self, 
                 metadata_file: str,
                 target_categories: List[str] = None,
                 sample_rate: int = 32000,
                 max_duration: float = 30.0,
                 device: str = "cuda"):
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.device = device
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_file)
        
        # Extract category from file path
        self.metadata['category'] = self.metadata['file_path'].apply(
            lambda x: x.split('/')[-2] if '/' in x else x.split('\\')[-2]
        )
        
        # Filter for target categories
        if target_categories:
            self.metadata = self.metadata[self.metadata['category'].isin(target_categories)]
        
        # Filter out missing files
        self.metadata = self.metadata[self.metadata['file_path'].apply(os.path.exists)]
        
        logger.info(f"📊 Loaded {len(self.metadata)} samples for categories: {target_categories}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        try:
            # Load audio with proper mono conversion
            audio_path = row['file_path']
            
            # Load audio and force mono conversion
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Ensure audio is 1D (mono)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0)  # Convert to mono by averaging channels
            
            # Trim to max duration
            max_samples = int(self.max_duration * self.sample_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            elif len(audio) < max_samples:
                audio = np.pad(audio, (0, max_samples - len(audio)), 'constant')
            
            # Get prompt
            prompt = row['prompt']
            
            return {
                'audio': torch.FloatTensor(audio),
                'prompt': prompt,
                'category': row['category'],
                'file_path': audio_path
            }
            
        except Exception as e:
            logger.error(f"❌ Error loading {row['file_path']}: {e}")
            # Return dummy data
            dummy_audio = torch.zeros(int(self.max_duration * self.sample_rate))
            return {
                'audio': dummy_audio,
                'prompt': "peaceful soundscape for relaxation",
                'category': "unknown",
                'file_path': "dummy"
            }

class LoRALayer(nn.Module):
    """LoRA layer for efficient fine-tuning."""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32.0, dropout: float = 0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA weights
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=2.236)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA."""
        return self.dropout(x @ self.lora_A.T) @ self.lora_B.T * self.scaling