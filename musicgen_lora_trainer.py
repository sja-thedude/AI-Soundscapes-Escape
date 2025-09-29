#!/usr/bin/env python3
"""
🎵 MusicGen + LoRA Training Script - Core Classes 🎵

This script contains the core training classes for MusicGen + LoRA fine-tuning.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import librosa
import logging
from typing import List
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
            audio_path = row['file_path']
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0)

            max_samples = int(self.max_duration * self.sample_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            elif len(audio) < max_samples:
                audio = np.pad(audio, (0, max_samples - len(audio)), 'constant')

            prompt = row['prompt']

            return {
                'audio': torch.FloatTensor(audio),
                'prompt': prompt,
                'category': row['category'],
                'file_path': audio_path
            }

        except Exception as e:
            logger.error(f"❌ Error loading {row['file_path']}: {e}")
            dummy_audio = torch.zeros(int(self.max_duration * self.sample_rate))
            return {
                'audio': dummy_audio,
                'prompt': "peaceful soundscape for relaxation",
                'category': "unknown",
                'file_path': "dummy"
            }


class LoRALayer(nn.Module):
    """LoRA Layer for Linear modules."""
    def __init__(self, in_features, out_features, rank=16, alpha=32.0, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.lora_A, mean=0.0, std=0.01)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        out = self.dropout(x_flat @ self.lora_A.T) @ self.lora_B.T
        out = out * self.scaling
        return out.reshape(*orig_shape[:-1], -1)