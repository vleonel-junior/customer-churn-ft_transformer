"""
Fonctions d'Entraînement pour FTT++ - Dataset Telecom

Ce module contient les fonctions d'entraînement, validation et évaluation
spécifiques au dataset Telecom. Ces fonctions sont passées en paramètre
aux gestionnaires d'entraînement de FTT++ pour maintenir la généricité.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Any, Callable
import zero


def create_loaders(y: Dict[str, torch.Tensor], batch_size: int, device: str) -> Tuple:
    """
    Crée les loaders d'entraînement et de validation.
    
    Args:
        y: Dict contenant les labels
        batch_size: Taille des batches
        device: Device à utiliser
        
    Returns:
        Tuple des loaders (train_loader, val_loader)
    """
    train_loader = zero.data.IndexLoader(len(y['train']), batch_size, device=device)
    val_loader = zero.data.IndexLoader(len(y['val']), batch_size, device=device)
    return train_loader, val_loader


def train(
    epoch: int,
    model,
    optimizer: torch.optim.Optimizer,
    X: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    y: Dict[str, torch.Tensor],
    train_loader,
    loss_fn: Callable
) -> float:
    """
    Fonction d'entraînement pour une époque.
    
    Args:
        epoch: Numéro de l'époque
        model: Modèle à entraîner
        optimizer: Optimiseur
        X: Données d'entrée
        y: Labels
        train_loader: Loader d'entraînement
        loss_fn: Fonction de perte
        
    Returns:
        Perte moyenne sur l'époque
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx in train_loader:
        optimizer.zero_grad()
        
        # Récupérer le batch
        x_num = X['train'][0][batch_idx]
        x_cat = X['train'][1][batch_idx] if X['train'][1] is not None else None
        y_batch = y['train'][batch_idx]
        
        # Forward pass
        logits, _ = model(x_num, x_cat)
        loss = loss_fn(logits.squeeze(-1), y_batch.float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def val(
    epoch: int,
    model,
    X: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    y: Dict[str, torch.Tensor],
    val_loader,
    loss_fn: Callable
) -> float:
    """
    Fonction de validation pour une époque.
    
    Args:
        epoch: Numéro de l'époque
        model: Modèle à évaluer
        X: Données d'entrée
        y: Labels
        val_loader: Loader de validation
        loss_fn: Fonction de perte
        
    Returns:
        Perte moyenne sur la validation
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx in val_loader:
            # Récupérer le batch
            x_num = X['val'][0][batch_idx]
            x_cat = X['val'][1][batch_idx] if X['val'][1] is not None else None
            y_batch = y['val'][batch_idx]
            
            # Forward pass
            logits, _ = model(x_num, x_cat)
            loss = loss_fn(logits.squeeze(-1), y_batch.float())
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, split: str, X: Dict, y: Dict, seed: int) -> Dict[str, Any]:
    """
    Évaluation complète du modèle sur un split donné.
    
    Args:
        model: Modèle à évaluer
        split: Split à évaluer ('val' ou 'test')
        X: Données d'entrée
        y: Labels
        seed: Seed pour la reproductibilité
        
    Returns:
        Dict avec les métriques de performance
    """
    model.eval()
    
    # Récupérer les données du split
    x_num = X[split][0]
    x_cat = X[split][1] if X[split][1] is not None else None
    y_true = y[split]
    
    # Prédictions
    with torch.no_grad():
        logits, _ = model(x_num, x_cat)
        y_pred_proba = torch.sigmoid(logits.squeeze(-1))
        y_pred = (y_pred_proba > 0.5).float()
    
    # Calculer les métriques
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    y_pred_proba_np = y_pred_proba.cpu().numpy()
    
    # Accuracy
    accuracy = (y_pred_np == y_true_np).mean()
    
    # AUC-ROC (si sklearn disponible)
    try:
        from sklearn.metrics import roc_auc_score
        auc_roc = roc_auc_score(y_true_np, y_pred_proba_np)
    except ImportError:
        auc_roc = None
    
    # Precision, Recall, F1 (si sklearn disponible)
    try:
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_np, y_pred_np, average='binary', zero_division=0
        )
    except ImportError:
        precision = recall = f1 = None
    
    # Loss
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss = loss_fn(logits.squeeze(-1), y_true.float()).item()
    
    results = {
        'accuracy': float(accuracy),
        'loss': float(loss),
        'n_samples': len(y_true),
        'split': split
    }
    
    # Ajouter les métriques sklearn si disponibles
    if auc_roc is not None:
        results['auc_roc'] = float(auc_roc)
    if precision is not None:
        results['precision'] = float(precision)
        results['recall'] = float(recall)
        results['f1'] = float(f1)
    
    return results


def apply_model(model, X: Dict, split: str = 'test') -> torch.Tensor:
    """
    Applique le modèle sur un split donné et retourne les probabilités.
    
    Args:
        model: Modèle à appliquer
        X: Données d'entrée
        split: Split à utiliser
        
    Returns:
        Probabilités prédites
    """
    model.eval()
    
    x_num = X[split][0]
    x_cat = X[split][1] if X[split][1] is not None else None
    
    with torch.no_grad():
        logits, _ = model(x_num, x_cat)
        proba = torch.sigmoid(logits.squeeze(-1))
    
    return proba