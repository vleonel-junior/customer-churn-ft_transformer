"""
Utilitaires pour les tâches de régression avec les modèles FT-Transformer.

Ce module contient les fonctions d'évaluation, de métriques et d'utilitaires
spécifiquement adaptées aux tâches de régression.
"""

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns


def regression_performance(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Calcule les métriques de performance pour la régression.
    
    Args:
        y_true: Valeurs vraies de forme (n_samples,)
        y_pred: Valeurs prédites de forme (n_samples,)
        
    Returns:
        tuple: (mse, rmse, mae, r2, mape)
            - mse: Mean Squared Error
            - rmse: Root Mean Squared Error  
            - mae: Mean Absolute Error
            - r2: Coefficient de détermination R²
            - mape: Mean Absolute Percentage Error
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (éviter la division par zéro)
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    return mse, rmse, mae, r2, mape


def regression_performance_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Retourne les métriques de régression sous forme de dictionnaire.
    
    Args:
        y_true: Valeurs vraies
        y_pred: Valeurs prédites
        
    Returns:
        dict: Dictionnaire avec les métriques de performance
    """
    mse, rmse, mae, r2, mape = regression_performance(y_true, y_pred)
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape)
    }


def plot_regression_results(y_true: np.ndarray, y_pred: np.ndarray, 
                          title: str = "Prédictions vs Valeurs Vraies",
                          save_path: str = None) -> None:
    """
    Crée un graphique de comparaison prédictions vs valeurs vraies.
    
    Args:
        y_true: Valeurs vraies
        y_pred: Valeurs prédites
        title: Titre du graphique
        save_path: Chemin pour sauvegarder le graphique (optionnel)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot prédictions vs vraies valeurs
    ax1.scatter(y_true, y_pred, alpha=0.6)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax1.set_xlabel('Valeurs Vraies')
    ax1.set_ylabel('Prédictions')
    ax1.set_title('Prédictions vs Valeurs Vraies')
    
    # Calcul des métriques pour affichage
    metrics = regression_performance_dict(y_true, y_pred)
    ax1.text(0.05, 0.95, f"R² = {metrics['r2']:.3f}\nRMSE = {metrics['rmse']:.3f}", 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Histogramme des résidus
    residuals = y_true - y_pred
    ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Résidus')
    ax2.set_ylabel('Fréquence')
    ax2.set_title('Distribution des Résidus')
    ax2.axvline(x=0, color='red', linestyle='--')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compute_feature_importance_regression(model, X_num, X_cat, feature_names=None):
    """
    Calcule l'importance des features pour un modèle de régression.
    
    Args:
        model: Modèle FTT+ pour la régression
        X_num: Features numériques
        X_cat: Features catégorielles
        feature_names: Noms des features (optionnel)
        
    Returns:
        dict: Importance des features
    """
    return model.get_cls_importance(X_num, X_cat, feature_names=feature_names)


def normalize_targets(y_train: torch.Tensor, y_val: torch.Tensor = None, y_test: torch.Tensor = None) -> Tuple[torch.Tensor, ...]:
    """
    Normalise les cibles pour améliorer l'entraînement de la régression.
    
    Args:
        y_train: Cibles d'entraînement
        y_val: Cibles de validation (optionnel)
        y_test: Cibles de test (optionnel)
        
    Returns:
        tuple: Tenseurs normalisés et statistiques (mean, std)
    """
    mean = y_train.mean()
    std = y_train.std()
    
    # Éviter la division par zéro
    std = torch.max(std, torch.tensor(1e-8))
    
    y_train_norm = (y_train - mean) / std
    
    results = [y_train_norm, mean, std]
    
    if y_val is not None:
        y_val_norm = (y_val - mean) / std
        results.append(y_val_norm)
        
    if y_test is not None:
        y_test_norm = (y_test - mean) / std
        results.append(y_test_norm)
    
    return tuple(results)


def denormalize_predictions(y_pred_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Dénormalise les prédictions normalisées.
    
    Args:
        y_pred_norm: Prédictions normalisées
        mean: Moyenne utilisée pour la normalisation
        std: Écart-type utilisé pour la normalisation
        
    Returns:
        torch.Tensor: Prédictions dénormalisées
    """
    return y_pred_norm * std + mean


def create_regression_dataset_example():
    """
    Crée un dataset d'exemple pour tester la régression.
    
    Returns:
        tuple: (X_num, X_cat, y, feature_names)
    """
    np.random.seed(42)
    
    # Features numériques
    n_samples = 1000
    X_num = np.random.randn(n_samples, 5)
    
    # Features catégorielles
    X_cat = np.random.randint(0, 3, size=(n_samples, 2))
    
    # Cible de régression (combinaison linéaire + bruit)
    y = (2 * X_num[:, 0] + 1.5 * X_num[:, 1] - 0.5 * X_num[:, 2] + 
         0.8 * X_cat[:, 0] + 0.3 * X_cat[:, 1] + 
         np.random.normal(0, 0.1, n_samples))
    
    feature_names = [
        'feature_num_0', 'feature_num_1', 'feature_num_2', 'feature_num_3', 'feature_num_4',
        'feature_cat_0', 'feature_cat_1'
    ]
    
    return (
        torch.tensor(X_num, dtype=torch.float32),
        torch.tensor(X_cat, dtype=torch.long),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        feature_names
    )


def print_regression_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Affiche les métriques de régression de manière formatée.
    
    Args:
        metrics: Dictionnaire des métriques
        prefix: Préfixe pour l'affichage
    """
    print(f"{prefix}Métriques de Régression:")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  R²:   {metrics['r2']:.6f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")


def get_regression_loss_function(loss_type: str = "mse") -> torch.nn.Module:
    """
    Retourne une fonction de perte pour la régression.
    
    Args:
        loss_type: Type de perte (uniquement 'mse' supporté pour la simplicité)
        
    Returns:
        torch.nn.Module: Fonction de perte MSE
    """
    if loss_type.lower() == "mse":
        return torch.nn.MSELoss()
    else:
        # Par défaut, utiliser MSE pour la simplicité
        print(f"Attention: type de perte '{loss_type}' non supporté, utilisation de MSE par défaut")
        return torch.nn.MSELoss()