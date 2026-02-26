"""
Code to assist fine tuning fog classif models for new/specific sites

Author: Joel Nicolow, Information and Computer Science, University of Hawaii at Manoa (Febuary 24, 2024)
"""

import torch
from torch.utils.data import Dataset
from glob import glob
import os
import numpy as np

from fogvision import fogimageclass

class SiteEmbeddingDataset(Dataset):
    def __init__(self, image_fns, basemodel):
        self.embeddings = []
        self.labels = []
        
        for fn in image_fns:
            ImageClass = fogimageclass.FogImage(fn)
            fog_val = ImageClass.fog_val  # 1 == fog, 0 == clear
            image_embedding = ImageClass.get_image_embedding(embedding_model=basemodel)
            
            self.embeddings.append(image_embedding)
            self.labels.append(fog_val)
        
        # Convert to tensors once upfront rather than every __getitem__ call
        self.embeddings = torch.tensor(np.array(self.embeddings), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
        print(f"Loaded {len(self.labels)} images | "
              f"Fog: {self.labels.sum().item()} | "
              f"Clear: {(self.labels == 0).sum().item()}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

