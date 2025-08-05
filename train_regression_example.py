"""
Script d'exemple pour entraîner le modèle Sparse FTT+ sur une tâche de régression.

Ce script montre comment utiliser le modèle InterpretableFTTPlusRegression
avec des données synthétiques pour une tâche de régression.
"""

import torch
import numpy as np
import os
from typing import Dict, Tuple
import time
import json

from sparse_ftt_plus_regression.model import InterpretableFTTPlusRegression
from utils_regression import (
    regression_performance_dict, print_regression_metrics, 
    plot_regression_results, normalize_targets, denormalize_predictions,
    create_regression_dataset_example, get_regression_loss_function
)


def split_data(X_num: torch.Tensor, X_cat: torch.Tensor, y: torch.Tensor, 
               train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict[str, Tuple[torch.Tensor, ...]]:
    """
    Divise les données en ensembles d'entraînement, validation et test.
    
    Args:
        X_num: Features numériques
        X_cat: Features catégorielles  
        y: Cibles
        train_ratio: Proportion d'entraînement
        val_ratio: Proportion de validation
        
    Returns:
        dict: Dictionnaire contenant les splits
    """
    n_samples = X_num.shape[0]
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    # Mélanger les indices
    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    return {
        'train': (X_num[train_indices], X_cat[train_indices], y[train_indices]),
        'val': (X_num[val_indices], X_cat[val_indices], y[val_indices]),
        'test': (X_num[test_indices], X_cat[test_indices], y[test_indices])
    }


def train_epoch(model, optimizer, data_loader, loss_fn, device):
    """
    Entraîne le modèle pour une époque.
    
    Args:
        model: Modèle à entraîner
        optimizer: Optimiseur
        data_loader: DataLoader pour l'entraînement
        loss_fn: Fonction de perte
        device: Device (CPU/GPU)
        
    Returns:
        float: Perte moyenne de l'époque
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for X_num_batch, X_cat_batch, y_batch in data_loader:
        X_num_batch = X_num_batch.to(device)
        X_cat_batch = X_cat_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(X_num_batch, X_cat_batch)
        loss = loss_fn(predictions, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate_model(model, data_loader, loss_fn, device):
    """
    Évalue le modèle sur un ensemble de données.
    
    Args:
        model: Modèle à évaluer
        data_loader: DataLoader pour l'évaluation
        loss_fn: Fonction de perte
        device: Device (CPU/GPU)
        
    Returns:
        tuple: (perte moyenne, prédictions, vraies valeurs)
    """
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    num_batches = 0
    
    with torch.no_grad():
        for X_num_batch, X_cat_batch, y_batch in data_loader:
            X_num_batch = X_num_batch.to(device)
            X_cat_batch = X_cat_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            pred_batch = model(X_num_batch, X_cat_batch)
            loss = loss_fn(pred_batch, y_batch)
            
            total_loss += loss.item()
            predictions.append(pred_batch.cpu())
            targets.append(y_batch.cpu())
            num_batches += 1
    
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    
    return total_loss / num_batches, predictions, targets


def create_data_loaders(data_splits: Dict, batch_size: int = 32):
    """
    Crée les DataLoaders pour l'entraînement.
    
    Args:
        data_splits: Dictionnaire avec les splits de données
        batch_size: Taille des batches
        
    Returns:
        dict: DataLoaders pour train/val/test
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    loaders = {}
    for split_name, (X_num, X_cat, y) in data_splits.items():
        dataset = TensorDataset(X_num, X_cat, y)
        shuffle = (split_name == 'train')
        loaders[split_name] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loaders


def main():
    """Fonction principale d'entraînement."""
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device: {device}")
    
    # Paramètres d'entraînement
    config = {
        'batch_size': 32,
        'n_epochs': 100,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'patience': 15,
        'loss_type': 'mse',  # 'mse', 'mae', 'huber', 'logcosh'
        'normalize_targets': True,
        'seed': 42
    }
    
    # Paramètres du modèle
    model_config = {
        'd_token': 128,
        'n_blocks': 3,
        'n_heads': 8,
        'attention_dropout': 0.1,
        'ffn_d_hidden': 256,
        'ffn_dropout': 0.1,
        'residual_dropout': 0.0,
        'd_out': 1,
        'attention_mode': 'hybrid'
    }
    
    print("Configuration:")
    print(f"  Modèle: {model_config}")
    print(f"  Entraînement: {config}")
    
    # Fixer la seed pour la reproductibilité
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Créer les données d'exemple
    print("\n=== Création des données d'exemple ===")
    X_num, X_cat, y, feature_names = create_regression_dataset_example()
    print(f"Données créées: {X_num.shape[0]} échantillons, {X_num.shape[1]} features numériques, {X_cat.shape[1]} features catégorielles")
    
    # Diviser les données
    data_splits = split_data(X_num, X_cat, y)
    print(f"Train: {data_splits['train'][0].shape[0]}, Val: {data_splits['val'][0].shape[0]}, Test: {data_splits['test'][0].shape[0]}")
    
    # Normaliser les cibles si demandé
    if config['normalize_targets']:
        y_train_norm, y_mean, y_std = normalize_targets(data_splits['train'][2])
        y_val_norm = (data_splits['val'][2] - y_mean) / y_std
        y_test_norm = (data_splits['test'][2] - y_mean) / y_std
        
        # Mettre à jour les splits avec les cibles normalisées
        data_splits['train'] = (data_splits['train'][0], data_splits['train'][1], y_train_norm)
        data_splits['val'] = (data_splits['val'][0], data_splits['val'][1], y_val_norm)
        data_splits['test'] = (data_splits['test'][0], data_splits['test'][1], y_test_norm)
        
        print("Cibles normalisées (mean={:.3f}, std={:.3f})".format(y_mean.item(), y_std.item()))
    
    # Créer les DataLoaders
    data_loaders = create_data_loaders(data_splits, config['batch_size'])
    
    # Créer le modèle
    print("\n=== Création du modèle ===")
    model = InterpretableFTTPlusRegression.make_baseline(
        n_num_features=X_num.shape[1],
        cat_cardinalities=[3, 3],  # Cardinalités pour les features catégorielles d'exemple
        **model_config
    )
    model.to(device)
    
    print(f"Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimiseur et fonction de perte
    optimizer = torch.optim.AdamW(model.optimization_param_groups(), 
                                  lr=config['lr'], weight_decay=config['weight_decay'])
    loss_fn = get_regression_loss_function(config['loss_type'])
    
    # Variables pour le suivi de l'entraînement
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    
    print(f"\n=== Début de l'entraînement ===")
    print(f"Fonction de perte: {config['loss_type']}")
    
    # Boucle d'entraînement
    for epoch in range(config['n_epochs']):
        start_time = time.time()
        
        # Entraînement
        train_loss = train_epoch(model, optimizer, data_loaders['train'], loss_fn, device)
        
        # Validation
        val_loss, val_predictions, val_targets = evaluate_model(model, data_loaders['val'], loss_fn, device)
        
        # Sauvegarde des métriques
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {epoch_time:.2f}s')
        
        # Early stopping et sauvegarde du meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f' <<< NOUVEAU MEILLEUR MODÈLE (val_loss: {val_loss:.6f})')
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f'\nEarly stopping à l\'époque {epoch} (patience: {config["patience"]})')
                break
    
    # Charger le meilleur modèle
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nMeilleur modèle chargé (époque {best_epoch}, val_loss: {best_val_loss:.6f})")
    
    # Évaluation finale
    print(f"\n=== Évaluation finale ===")
    
    # Évaluation sur validation
    val_loss, val_predictions, val_targets = evaluate_model(model, data_loaders['val'], loss_fn, device)
    
    # Évaluation sur test
    test_loss, test_predictions, test_targets = evaluate_model(model, data_loaders['test'], loss_fn, device)
    
    # Dénormaliser les prédictions si nécessaire
    if config['normalize_targets']:
        val_predictions = denormalize_predictions(val_predictions, y_mean, y_std)
        val_targets = denormalize_predictions(val_targets, y_mean, y_std)
        test_predictions = denormalize_predictions(test_predictions, y_mean, y_std)
        test_targets = denormalize_predictions(test_targets, y_mean, y_std)
    
    # Calculer les métriques
    val_metrics = regression_performance_dict(val_targets.numpy(), val_predictions.numpy())
    test_metrics = regression_performance_dict(test_targets.numpy(), test_predictions.numpy())
    
    print("\n--- Résultats sur Validation ---")
    print_regression_metrics(val_metrics)
    
    print("\n--- Résultats sur Test ---")
    print_regression_metrics(test_metrics)
    
    # Analyse d'interprétabilité
    print(f"\n=== Analyse d'interprétabilité ===")
    model.eval()
    with torch.no_grad():
        # Prendre un échantillon des données de test
        sample_X_num = data_splits['test'][0][:10].to(device)
        sample_X_cat = data_splits['test'][1][:10].to(device)
        
        importance = model.get_cls_importance(sample_X_num, sample_X_cat, feature_names=feature_names)
        
        print("Importance des features (attention CLS):")
        for feature_name, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature_name}: {score:.4f}")
    
    # Convertir les tenseurs en float Python pour la sérialisation JSON
    def convert_to_python(obj):
        if hasattr(obj, 'item'):  # Tenseurs scalaires
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python(v) for v in obj]
        return obj
    
    # Sauvegarder les résultats
    results = {
        'config': config,
        'model_config': model_config,
        'training_results': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss
        },
        'performance_results': {
            'val': convert_to_python(val_metrics),
            'test': convert_to_python(test_metrics)
        },
        'feature_importance': convert_to_python(importance)
    }
    
    # Créer le dossier de sortie
    output_dir = "results_regression_example"
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder les résultats
    with open(f'{output_dir}/results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Sauvegarder le modèle
    torch.save(model.state_dict(), f'{output_dir}/best_model.pt')
    
    print(f"\nRésultats sauvegardés dans '{output_dir}/'")
    print("Entraînement terminé!")


if __name__ == '__main__':
    main()