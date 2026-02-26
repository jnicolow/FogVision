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
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score, precision_recall_curve

from fogvision import fogimageclass

class SiteEmbeddingDataset(Dataset):
    def __init__(self, image_fns, basemodel, disable_tqdm:bool=True):
        self.embeddings = []
        self.labels = []
        
        for fn in tqdm(image_fns, disable=disable_tqdm):
            ImageClass = fogimageclass.FogImage(fn, crop_size=256)
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
    


def get_balanced_train_indices(train_pool_indices, dataset, n):
    """Take n indices with balanced fog/clear from the train pool."""
    fog_idx = [i for i in train_pool_indices if dataset[i][1] == 1]
    clear_idx = [i for i in train_pool_indices if dataset[i][1] == 0]
    
    n_per_class = n // 2
    # Take first n_per_class from each class
    selected = fog_idx[:n_per_class] + clear_idx[:n_per_class]
    return selected


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


def find_optimal_threshold(model, calib_loader, device, objective='balanced'):
    """Find threshold that maximises accuracy on a calibration set."""
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for embeddings, labels in calib_loader:
            outputs = model(embeddings.to(device))
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels, all_probs = np.array(all_labels), np.array(all_probs)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    if objective == 'balanced':
        # current: maximise tpr - fpr (Youden J)
        idx = np.argmax(tpr - fpr)
    elif objective == 'f1':
        # maximise F1 directly
        precision, recall, thresh = precision_recall_curve(all_labels, all_probs)
        idx = np.argmax(2 * precision * recall / (precision + recall + 1e-8))
        return float(thresh[idx])
    
    return float(thresholds[idx]) 


def finetune_and_evaluate(model, train_loader, test_loader, device,
                           epochs=20, lr=1e-3, layers_to_unfreeze='all', calib_loader=None):
    model = copy.deepcopy(model)
    model.to(device)

    if layers_to_unfreeze == 'last_only':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc6.parameters():
            param.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(embeddings), labels)
            loss.backward()
            optimizer.step()

    # Find optimal threshold from calibration set if provided, else use 0.5
    threshold = find_optimal_threshold(model, calib_loader, device) if calib_loader else 0.5

    return evaluate(model, test_loader, device, threshold=threshold)


from sklearn.metrics import roc_curve


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs >= threshold).long()  # use threshold instead of argmax
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    return {
        'auroc':     roc_auc_score(all_labels, all_probs),
        'accuracy':  accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'f1':        f1_score(all_labels, all_preds, zero_division=0),
    }