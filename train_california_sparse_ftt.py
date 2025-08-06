"""
Script d'entraînement du modèle Sparse FTT+ sur le dataset California Housing.
"""

import torch
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple
import time

from sparse_ftt_plus_regression.model import InterpretableFTTPlusRegression
from utils_regression import (
    regression_performance_dict, print_regression_metrics, 
    normalize_targets, denormalize_predictions,
    get_regression_loss_function
)
from interpretability_analyzer import analyze_interpretability


def load_california_housing():
    """Charge et prépare le dataset California Housing."""
    data = fetch_california_housing()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    # Standardiser les features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convertir en tenseurs PyTorch
    X_num = torch.FloatTensor(X_scaled)
    X_cat = None  # Pas de features catégorielles pour California Housing
    y = torch.FloatTensor(y)
    
    return X_num, X_cat, y, list(feature_names)


def split_data(X_num: torch.Tensor, X_cat, y: torch.Tensor, 
               train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict[str, Tuple]:
    """Divise les données en ensembles d'entraînement, validation et test."""
    n_samples = X_num.shape[0]
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    return {
        'train': (X_num[train_indices], X_cat, y[train_indices]),
        'val': (X_num[val_indices], X_cat, y[val_indices]),
        'test': (X_num[test_indices], X_cat, y[test_indices])
    }


def create_data_loaders(data_splits: Dict, batch_size: int = 256):
    """Crée les DataLoaders pour l'entraînement."""
    from torch.utils.data import TensorDataset, DataLoader
    
    loaders = {}
    for split_name, (X_num, X_cat, y) in data_splits.items():
        # Créer un placeholder pour X_cat si None
        X_cat_placeholder = torch.zeros(X_num.shape[0], 0, dtype=torch.long) if X_cat is None else X_cat
        dataset = TensorDataset(X_num, X_cat_placeholder, y)
        shuffle = (split_name == 'train')
        loaders[split_name] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loaders


def train_epoch(model, optimizer, data_loader, loss_fn, device):
    """Entraîne le modèle pour une époque."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for X_num_batch, X_cat_batch, y_batch in data_loader:
        X_num_batch = X_num_batch.to(device)
        # Passer None pour x_cat au lieu du tenseur vide
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        predictions = model(X_num_batch, None)  # x_cat = None
        # Aplatir les prédictions pour correspondre aux targets
        predictions = predictions.squeeze(-1) if predictions.dim() > 1 else predictions
        loss = loss_fn(predictions, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate_model(model, data_loader, loss_fn, device):
    """Évalue le modèle sur un ensemble de données."""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    num_batches = 0
    
    with torch.no_grad():
        for X_num_batch, X_cat_batch, y_batch in data_loader:
            X_num_batch = X_num_batch.to(device)
            # Passer None pour x_cat au lieu du tenseur vide
            y_batch = y_batch.to(device)
            
            pred_batch = model(X_num_batch, None)  # x_cat = None
            # Aplatir les prédictions pour correspondre aux targets
            pred_batch = pred_batch.squeeze(-1) if pred_batch.dim() > 1 else pred_batch
            loss = loss_fn(pred_batch, y_batch)
            
            total_loss += loss.item()
            predictions.append(pred_batch.cpu())
            targets.append(y_batch.cpu())
            num_batches += 1
    
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    
    return total_loss / num_batches, predictions, targets


def main():
    """Fonction principale d'entraînement."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Sparse FTT+ - California Housing ===")
    print(f"Device: {device}")
    
    # Configuration
    config = {
        'batch_size': 256,
        'n_epochs': 50,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'patience': 15,
        'loss_type': 'mse',
        'normalize_targets': True,
        'seed': 42
    }
    
    # Paramètres du modèle Sparse FTT+
    model_config = {
        'd_token': 64,
        'n_blocks': 3,
        'n_heads': 8,
        'attention_dropout': 0.1,
        'ffn_d_hidden': 256,
        'ffn_dropout': 0.1,
        'residual_dropout': 0.0,
        'd_out': 1,
        'attention_mode': 'hybrid'
        # Suppression de 'average_layer': True car non supporté par Sparse FTT+
    }
    
    print(f"Configuration: {config}")
    print(f"Modèle: {model_config}")
    
    # Fixer la seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Charger les données
    print("\n=== Chargement des données ===")
    X_num, X_cat, y, feature_names = load_california_housing()
    print(f"Données: {X_num.shape[0]} échantillons, {X_num.shape[1]} features")
    
    # Diviser les données
    data_splits = split_data(X_num, X_cat, y)
    print(f"Train: {data_splits['train'][0].shape[0]}, Val: {data_splits['val'][0].shape[0]}, Test: {data_splits['test'][0].shape[0]}")
    
    # Normaliser les cibles
    if config['normalize_targets']:
        y_train_norm, y_mean, y_std = normalize_targets(data_splits['train'][2])
        y_val_norm = (data_splits['val'][2] - y_mean) / y_std
        y_test_norm = (data_splits['test'][2] - y_mean) / y_std
        
        data_splits['train'] = (data_splits['train'][0], data_splits['train'][1], y_train_norm)
        data_splits['val'] = (data_splits['val'][0], data_splits['val'][1], y_val_norm)
        data_splits['test'] = (data_splits['test'][0], data_splits['test'][1], y_test_norm)
        
        print(f"Cibles normalisées (mean={y_mean.item():.3f}, std={y_std.item():.3f})")
    
    # Créer les DataLoaders
    data_loaders = create_data_loaders(data_splits, config['batch_size'])
    
    # Créer le modèle
    print("\n=== Création du modèle Sparse FTT+ ===")
    model = InterpretableFTTPlusRegression.make_baseline(
        n_num_features=X_num.shape[1],
        cat_cardinalities=[],
        **model_config
    )
    model.to(device)
    
    print(f"Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimiseur et fonction de perte
    optimizer = torch.optim.AdamW(model.optimization_param_groups(), 
                                  lr=config['lr'], weight_decay=config['weight_decay'])
    loss_fn = get_regression_loss_function(config['loss_type'])
    
    # Variables pour le suivi
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    
    print(f"\n=== Entraînement ({config['n_epochs']} époques max) ===")
    
    # Boucle d'entraînement
    for epoch in range(config['n_epochs']):
        start_time = time.time()
        
        train_loss = train_epoch(model, optimizer, data_loaders['train'], loss_fn, device)
        val_loss, val_predictions, val_targets = evaluate_model(model, data_loaders['val'], loss_fn, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {epoch_time:.2f}s')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f' <<< NOUVEAU MEILLEUR MODÈLE')
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f'\nEarly stopping à l\'époque {epoch}')
                break
    
    # Charger le meilleur modèle
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Évaluation finale
    print(f"\n=== Évaluation finale ===")
    
    val_loss, val_predictions, val_targets = evaluate_model(model, data_loaders['val'], loss_fn, device)
    test_loss, test_predictions, test_targets = evaluate_model(model, data_loaders['test'], loss_fn, device)
    
    # Dénormaliser
    if config['normalize_targets']:
        val_predictions = denormalize_predictions(val_predictions, y_mean, y_std)
        val_targets = denormalize_predictions(val_targets, y_mean, y_std)
        test_predictions = denormalize_predictions(test_predictions, y_mean, y_std)
        test_targets = denormalize_predictions(test_targets, y_mean, y_std)
    
    # Calculer les métriques
    val_metrics = regression_performance_dict(val_targets.numpy(), val_predictions.numpy())
    test_metrics = regression_performance_dict(test_targets.numpy(), test_predictions.numpy())
    
    print("\n--- Résultats Validation ---")
    print_regression_metrics(val_metrics)
    
    print("\n--- Résultats Test ---")
    print_regression_metrics(test_metrics)
    
    # Analyse d'interprétabilité
    print(f"\n=== Analyse d'interprétabilité ===")
    
    X_dict = {'test': (data_splits['test'][0], data_splits['test'][1])}
    y_dict = {'test': data_splits['test'][2]}
    
    performance_results = {
        'val': list(val_metrics.values()),
        'test': list(test_metrics.values())
    }
    
    training_results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'total_epochs': len(train_losses)
    }
    
    interpretability_results = analyze_interpretability(
        model=model,
        X=X_dict,
        y=y_dict,
        model_name="sparse_ftt_plus_regression",
        seed=config['seed'],
        model_config=model_config,
        training_results=training_results,
        performance_results=performance_results,
        feature_names=feature_names,
        task_type='regression',  # Spécifier le type de tâche
        results_base_dir="results"
    )
    
    print("=== Sparse FTT+ terminé ===")


if __name__ == '__main__':
    main()
