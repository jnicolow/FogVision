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
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score, precision_recall_curve

from fogvision import fogimageclass

class SiteEmbeddingDataset(Dataset):
    def __init__(self, image_fns, basemodel, disable_tqdm:bool=True):
        self.embeddings = []
        self.labels = []
        self.image_fns = image_fns
        
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
        self.fns = data.get('fns', [None] * len(self.labels))  # graceful fallback for old saves
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]  # fns accessed directly via dataset.fns[idx]


def save_test_image_plots(dataset, test_indices, model, device, output_dir, sitename, DoN, 
                           default_thresh=0.5, optimal_thresh=0.5):
    model = copy.deepcopy(model).cpu()
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    for idx in test_indices:
        embedding, label = dataset[idx]
        fn = dataset.fns[idx]
        
        with torch.no_grad():
            prob = torch.softmax(model(embedding.unsqueeze(0).cpu()), dim=1)[0, 1].item()

        default_pred = 'fog' if prob >= default_thresh else 'clear'
        optimal_pred = 'fog' if prob >= optimal_thresh else 'clear'
        true_label   = 'fog' if label == 1 else 'clear'
        if default_pred == optimal_pred and optimal_pred == true_label:
            continue

        default_correct = default_pred == true_label
        optimal_correct = optimal_pred == true_label

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.imshow(plt.imread(fn))
        ax.axis('off')

        # Border color driven by optimal threshold correctness
        border_color = '#2ecc71' if optimal_correct else '#e74c3c'
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(5)

        # True label top left — always white
        ax.text(0.02, 0.97, f'True: {true_label}  |  prob: {prob:.2f}',
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                va='top', ha='left', color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#333333', alpha=0.8))

        # Default threshold prediction
        ax.text(0.02, 0.82, f'Default (t=0.50): {default_pred}',
                transform=ax.transAxes, fontsize=9,
                va='top', ha='left', color='white',
                bbox=dict(boxstyle='round,pad=0.3', 
                          facecolor='#2ecc71' if default_correct else '#e74c3c', 
                          alpha=0.9))

        # Optimal threshold prediction
        ax.text(0.02, 0.68, f'Optimal (t={optimal_thresh:.2f}): {optimal_pred}',
                transform=ax.transAxes, fontsize=9,
                va='top', ha='left', color='white',
                bbox=dict(boxstyle='round,pad=0.3', 
                          facecolor='#2ecc71' if optimal_correct else '#e74c3c', 
                          alpha=0.9))

        plt.tight_layout()
        img_name = os.path.splitext(os.path.basename(fn))[0]
        save_path = os.path.join(output_dir, f'{sitename}_{DoN}_{img_name}.jpg')
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close()


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


def evaluate_with_probs(model, loader, device, threshold=0.5):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs >= threshold).long()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    metrics = {
        'auroc':     roc_auc_score(all_labels, all_probs),
        'accuracy':  accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'f1':        f1_score(all_labels, all_preds, zero_division=0),
    }
    return metrics, all_probs, all_labels