"""
Module d'Analyse d'Interprétabilité Générique

Centralise l'analyse d'interprétabilité pour tous les modèles FTT de manière modulaire.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Importation directe des fonctions de visualisation
import visualisation


class InterpretabilityAnalyzer:
    """Analyseur d'interprétabilité générique et modulaire pour modèles FTT."""
    
    def __init__(self, results_base_dir: str = 'results'):
        """
        Initialise l'analyseur.
        
        Args:
            results_base_dir: Répertoire de base pour les résultats.
        """
        self.results_dir = Path(results_base_dir)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Crée les répertoires nécessaires."""
        for subdir in ['métriques', 'heatmaps', 'best_models']:
            (self.results_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def analyze_and_save(
        self,
        model: torch.nn.Module,
        X: Dict[str, Union[np.ndarray, torch.Tensor]],
        y: Dict[str, Union[np.ndarray, torch.Tensor]],
        model_name: str,
        seed: int,
        model_config: Dict[str, Any],
        training_results: Dict[str, Any],
        performance_results: Dict[str, List[float]],
        feature_names: List[str],
        task_type: str = 'classification',
        local_output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyse complète d'interprétabilité avec sauvegarde.
        
        Args:
            model: Le modèle PyTorch entraîné.
            X: Dictionnaire contenant les données, e.g., {'test': (x_num, x_cat)}.
            y: Dictionnaire contenant les étiquettes.
            model_name: Nom du modèle.
            seed: Graine aléatoire utilisée.
            model_config: Configuration du modèle.
            training_results: Résultats de l'entraînement.
            performance_results: Résultats de performance (val, test).
            feature_names: Liste des noms de features.
            task_type: Type de tâche ('classification' ou 'regression').
            local_output_dir: Répertoire de sauvegarde locale (optionnel).
            
        Returns:
            dict: Résultats bruts de l'analyse d'interprétabilité.
        """
        if not feature_names:
            raise ValueError("feature_names est obligatoire")
        
        if task_type not in ['classification', 'regression']:
            raise ValueError("task_type doit être 'classification' ou 'regression'")
        
        print(f"\n=== Analyse d'interprétabilité ({task_type}) ===")
        
        # 1. Calcul de l'importance des features via le token CLS
        model.eval()
        with torch.no_grad():
            cls_importance = model.get_cls_importance(X['test'][0], X['test'][1], feature_names)
        
        # 2. Affichage du top 10 des features importantes
        sorted_importance = sorted(cls_importance.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 10 features importantes:")
        for i, (feature, score) in enumerate(sorted_importance[:10], 1):
            print(f"  {i:2d}. {feature:<20}: {score:.4f}")
        
        # 3. Sauvegarde des métriques de performance
        self._save_metrics(model_name, seed, model_config, training_results, performance_results, model, task_type)
        
        # 4. Sauvegarde des résultats bruts d'interprétabilité
        interpretability_results = self._save_interpretability(model_name, seed, feature_names, cls_importance, sorted_importance)
        
        # 5. Génération et sauvegarde des visualisations
        self._generate_and_save_visualizations(model, X, feature_names, model_name, seed, cls_importance)
        
        # 6. Sauvegarde des poids du modèle
        self._save_model(model_name, seed, model)
        
        # 7. Résumé des fichiers sauvegardés
        self._print_summary(model_name, seed)
        
        return interpretability_results
    
    def _save_metrics(self, model_name: str, seed: int, model_config: Dict, training_results: Dict, performance_results: Dict, model: torch.nn.Module, task_type: str):
        """Sauvegarde les métriques de performance selon le type de tâche."""
        
        # Définition des noms de métriques selon le type de tâche
        if task_type == 'classification':
            metric_names = [
                "roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "mcc",
                "sensitivity", "specificity", "precision", "f1", "cohen_kappa"
            ]
        else:  # regression
            metric_names = [
                "mse", "rmse", "mae", "r2", "mape"
            ]
        
        def to_named_dict(values):
            if isinstance(values, dict):
                return values
            return {name: float(val) for name, val in zip(metric_names, values)}
        
        data = {
            'model_name': model_name,
            'seed': seed,
            'task_type': task_type,
            'model_config': model_config,
            'training': {**training_results, 'n_parameters': sum(p.numel() for p in model.parameters())},
            'performance': {
                'val': to_named_dict(performance_results['val']),
                'test': to_named_dict(performance_results['test'])
            }
        }
        
        path = self.results_dir / 'métriques' / f'{model_name}_metrics_seed_{seed}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_interpretability(self, model_name: str, seed: int, feature_names: List[str], cls_importance: Dict[str, float], sorted_importance: List) -> Dict:
        """Sauvegarde les résultats bruts d'interprétabilité."""
        data = {
            'model_name': model_name,
            'seed': seed,
            'feature_names': feature_names,
            'cls_importance': {k: float(v) for k, v in cls_importance.items()},
            'top_features': [{'name': name, 'importance': float(score)} for name, score in sorted_importance[:10]]
        }
        
        path = self.results_dir / 'métriques' / f'{model_name}_importance_seed_{seed}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return data
    
    def _generate_and_save_visualizations(self, model: torch.nn.Module, X: Dict, feature_names: List[str], model_name: str, seed: int, cls_importance: Dict[str, float]):
        """Génère et sauvegarde les visualisations de manière directe et modulaire."""
        print("\nGénération des visualisations...")
        
        # 1. Graphique d'importance des features
        importance_output_path = str(self.results_dir / 'heatmaps' / f'{model_name}_importance_seed_{seed}.png')
        visualisation.create_importance_bar_chart(
            cls_importance,
            feature_names=feature_names,
            output_path=importance_output_path,
            title=f"Importance - {model_name.replace('_', ' ').title()} (Seed {seed})"
        )
        
        # 2. Heatmap d'attention complète (si le modèle la supporte)
        # On vérifie si le modèle a la méthode get_full_attention_matrix
        if hasattr(model, 'get_full_attention_matrix'):
            try:
                with torch.no_grad():
                    full_attention_matrix = model.get_full_attention_matrix(X['test'][0], X['test'][1])
                
                attention_output_path = str(self.results_dir / 'heatmaps' / f'{model_name}_attention_seed_{seed}.png')
                visualisation.visualize_full_interactions(
                    full_attention_matrix.cpu().numpy(),
                    feature_names=feature_names,
                    output_path=attention_output_path,
                    title=f'Attention - {model_name} (Seed {seed})'
                )
            except Exception as e:
                print(f"Avertissement: Impossible de générer la heatmap d'attention pour {model_name}: {e}")
        else:
            print(f"Info: Le modèle {model_name} ne supporte pas la heatmap d'attention complète (methode get_full_attention_matrix manquante).")
    
    def _save_model(self, model_name: str, seed: int, model: torch.nn.Module):
        """Sauvegarde les poids du modèle."""
        path = self.results_dir / 'best_models' / f'{model_name}_weights_seed_{seed}.pt'
        torch.save(model.state_dict(), path)
    
    
    def _print_summary(self, model_name: str, seed: int):
        """Affiche un résumé des fichiers sauvegardés."""
        print(f"\n=== Résultats sauvegardés ===")
        base = f"{self.results_dir}/{model_name}_*_seed_{seed}"
        print(f"Métriques: {base.replace('*', 'metrics')}.json")
        print(f"Importance: {base.replace('*', 'importance')}.json")
        print(f"Visualisations: {base.replace('*', '{importance|attention}')}.png")
        print(f"Modèle: {base.replace('*', 'weights')}.pt")


def analyze_interpretability(
    model: torch.nn.Module,
    X: Dict[str, Union[np.ndarray, torch.Tensor]],
    y: Dict[str, Union[np.ndarray, torch.Tensor]],
    model_name: str,
    seed: int,
    model_config: Dict[str, Any],
    training_results: Dict[str, Any],
    performance_results: Dict[str, List[float]],
    feature_names: List[str],
    task_type: Optional[str] = None,
    local_output_dir: Optional[str] = None,
    results_base_dir: str = 'results'
) -> Dict[str, Any]:
    """
    Fonction d'analyse d'interprétabilité générique et modulaire.
    
    Args:
        model: Le modèle PyTorch entraîné.
        X: Dictionnaire contenant les données.
        y: Dictionnaire contenant les étiquettes.
        model_name: Nom du modèle.
        seed: Graine aléatoire utilisée.
        model_config: Configuration du modèle.
        training_results: Résultats de l'entraînement.
        performance_results: Résultats de performance (val, test).
        feature_names: Liste des noms de features.
        task_type: Type de tâche ('classification', 'regression', ou None pour comportement par défaut).
        local_output_dir: Répertoire de sauvegarde locale (optionnel).
        results_base_dir: Répertoire de base pour les résultats.
        
    Returns:
        dict: Résultats bruts de l'analyse d'interprétabilité.
    """
    
    # Si task_type n'est pas spécifié, utiliser 'classification' par défaut
    if task_type is None:
        task_type = 'classification'
        print(f"task_type non spécifié, utilisation de 'classification' par défaut")
    
    analyzer = InterpretabilityAnalyzer(results_base_dir)
    return analyzer.analyze_and_save(model, X, y, model_name, seed, model_config, training_results, performance_results, feature_names, task_type, local_output_dir)