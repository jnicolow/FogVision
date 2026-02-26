"""
Code to assist fine tuning fog classif models for new/specific sites

Author: Joel Nicolow, Information and Computer Science, University of Hawaii at Manoa (Febuary 24, 2024)
"""

from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from glob import glob
import os
import numpy as np
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score

from fogvision import fogimageclass

class SiteEmbeddingDataset(Dataset):
    def __init__(self, image_fns, basemodel, disable_tqdm:bool=True):
        self.embeddings = []
        self.labels = []
        
        for fn in tqdm(image_fns, disable=disable_tqdm):
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



class PrecomputedEmbeddingDataset(Dataset):
    def __init__(self, filepath):
        data = torch.load(filepath)
        self.embeddings = data['embeddings']
        self.labels = data['labels']
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
    




def get_balanced_test_indices(dataset, n_per_class=10):
    """Get last n_per_class fog and last n_per_class clear indices."""
    fog_indices = [i for i in range(len(dataset)) if dataset[i][1] == 1]
    clear_indices = [i for i in range(len(dataset)) if dataset[i][1] == 0]
    
    test_indices = fog_indices[-n_per_class:] + clear_indices[-n_per_class:]
    test_indices_set = set(test_indices)
    train_pool_indices = [i for i in range(len(dataset)) if i not in test_indices_set]
    
    return test_indices, train_pool_indices


def evaluate(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels, all_preds, all_probs = np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    return {
        'auroc':     roc_auc_score(all_labels, all_probs),
        'accuracy':  accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'f1':        f1_score(all_labels, all_preds, zero_division=0),
    }


def finetune_and_evaluate(model, train_loader, test_loader, device, epochs=20, lr=1e-4):
    """Fine-tune model and return eval metrics. Works on a copy so original is unchanged."""
    model = copy.deepcopy(model)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(embeddings), labels)
            loss.backward()
            optimizer.step()
    
    return evaluate(model, test_loader, device)

