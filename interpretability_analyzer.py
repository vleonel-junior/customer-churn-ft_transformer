"""
Module d'Analyse d'Interprétabilité Générique

Ce module centralise toutes les fonctionnalités d'analyse d'interprétabilité
pour tous les modèles FTT (ftt_plus, ftt_plus_plus, ftt_random, etc.).
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from ftt_plus.visualisation import create_importance_bar_chart


class InterpretabilityAnalyzer:
    """Analyseur d'interprétabilité générique pour modèles FTT interprétables."""
    
    def __init__(self, results_base_dir: str = 'results/results_telecom'):
        """
        Args:
            results_base_dir: répertoire de base pour sauvegarder les résultats
        """
        self.results_dir = Path(results_base_dir)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Crée les répertoires nécessaires s'ils n'existent pas."""
        (self.results_dir / 'métriques').mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'heatmaps').mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'best_models').mkdir(parents=True, exist_ok=True)
    
    def analyze_and_save(
        self,
        model,
        X: Dict,
        y: Dict,
        model_name: str,
        seed: int,
        model_config: Dict[str, Any],
        training_results: Dict[str, Any],
        performance_results: Dict[str, List[float]],
        feature_names: Optional[List[str]] = None,
        local_output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyse complète d'interprétabilité avec sauvegarde des résultats.
        
        Args:
            model: modèle FTT entraîné (doit avoir get_cls_importance)
            X: données d'entrée (dict avec 'train', 'val', 'test')
            y: labels (dict avec 'train', 'val', 'test')
            model_name: nom du modèle (ex: 'ftt_plus', 'ftt_plus_plus', 'ftt_random')
            seed: seed utilisé pour l'entraînement
            model_config: configuration du modèle
            training_results: résultats d'entraînement (losses, epochs, etc.)
            performance_results: métriques de performance (val et test)
            feature_names: noms des features (par défaut généré automatiquement)
            local_output_dir: répertoire de sauvegarde locale (optionnel)
            
        Returns:
            dict: résultats d'interprétabilité calculés
        """
        print("\n=== Analyse d'interprétabilité ===")
        
        # Noms des features par défaut si non fournis
        if feature_names is None:
            feature_names = self._get_default_telecom_feature_names()
        
        # Calcul de l'importance sur l'ensemble de test
        print("Calcul de l'importance des features sur l'ensemble de test...")
        model.eval()
        with torch.no_grad():
            cls_importance = model.get_cls_importance(
                X['test'][0], X['test'][1], feature_names
            )
        
        # Affichage des features les plus importantes
        print("\nTop 10 des features les plus importantes:")
        sorted_importance = sorted(cls_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_importance[:10], 1):
            print(f"  {i:2d}. {feature:<20}: {importance:.4f}")
        
        # Sauvegarde des métriques de performance
        self._save_performance_metrics(
            model_name, seed, model_config, training_results, performance_results, model
        )
        
        # Sauvegarde des résultats d'interprétabilité
        interpretability_results = self._save_interpretability_metrics(
            model_name, seed, feature_names, cls_importance, sorted_importance
        )
        
        # Génération du graphique d'importance
        self._generate_importance_plot(model_name, seed, cls_importance)
        
        # Génération de la heatmap d'attention pour tous les modèles FTT
        self._generate_attention_heatmap(model, X, feature_names, model_name, seed)
        
        # Sauvegarde du modèle
        self._save_model(model_name, seed, model)
        
        # Sauvegarde locale si demandée
        if local_output_dir:
            self._save_local_results(
                local_output_dir, training_results, performance_results, cls_importance, model
            )
        
        self._print_summary(model_name, seed)
        
        return interpretability_results
    
    def _get_default_telecom_feature_names(self) -> List[str]:
        """Retourne les noms de features par défaut pour le dataset telecom."""
        return [
            'tenure', 'MonthlyCharges', 'TotalCharges',  # Features numériques
            'gender', 'SeniorCitizen', 'Partner', 'Dependents',  # Features catégorielles
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod'
        ]
    
    def _save_performance_metrics(
        self,
        model_name: str,
        seed: int,
        model_config: Dict[str, Any],
        training_results: Dict[str, Any],
        performance_results: Dict[str, List[float]],
        model
    ):
        """Sauvegarde les métriques de performance."""
        val_perf, test_perf = performance_results['val'], performance_results['test']
        
        metrics_names = [
            'roc_auc', 'pr_auc', 'accuracy', 'balanced_accuracy', 'mcc',
            'sensitivity', 'specificity', 'precision', 'f1', 'cohen_kappa'
        ]
        
        performance_data = {
            'model_name': model_name,
            'seed': seed,
            'model_config': model_config,
            'training': {
                **training_results,
                'n_parameters': sum(p.numel() for p in model.parameters())
            },
            'performance': {
                'val_performance': dict(zip(metrics_names, [float(x) for x in val_perf])),
                'test_performance': dict(zip(metrics_names, [float(x) for x in test_perf]))
            }
        }
        
        metrics_path = self.results_dir / 'métriques' / f'{model_name}_model_performance_metrics_seed_{seed}.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(performance_data, f, indent=2, ensure_ascii=False)
    
    def _save_interpretability_metrics(
        self,
        model_name: str,
        seed: int,
        feature_names: List[str],
        cls_importance: Dict[str, float],
        sorted_importance: List[tuple]
    ) -> Dict[str, Any]:
        """Sauvegarde les résultats d'interprétabilité."""
        interpretability_data = {
            'model_name': model_name,
            'seed': seed,
            'feature_names': feature_names,
            'cls_importance': {k: float(v) for k, v in cls_importance.items()},
            'top_features': [
                {'name': name, 'importance': float(importance)}
                for name, importance in sorted_importance[:10]
            ]
        }
        
        interpretability_path = self.results_dir / 'métriques' / f'{model_name}_feature_importance_analysis_seed_{seed}.json'
        with open(interpretability_path, 'w', encoding='utf-8') as f:
            json.dump(interpretability_data, f, indent=2, ensure_ascii=False)
        
        return interpretability_data
    
    def _generate_importance_plot(self, model_name: str, seed: int, cls_importance: Dict[str, float]):
        """Génère le graphique d'importance des features."""

        print("\nGénération du graphique d'importance des features...")
        output_path = self.results_dir / 'heatmaps' / f'{model_name}_feature_importance_chart_seed_{seed}.png'
        
        # Titre personnalisé selon le modèle
        model_titles = {
            'interpretable_ftt_plus': 'FTT+ Interprétable',
            'interpretable_ftt_plus_plus': 'FTT++ Interprétable'
        }
        title = f"Importance des Features - {model_titles.get(model_name, model_name.upper())} (Seed {seed})"
        
        # Utiliser la visualisation appropriée selon le modèle
        if 'ftt_plus_plus' in model_name:
            # Pour modèle FTT++ :
            try:
                from ftt_plus_plus.visualisation import create_ftt_plus_plus_importance_chart
                create_ftt_plus_plus_importance_chart(
                    cls_importance,
                    output_path=str(output_path),
                    title=title
                )
            except ImportError:
                print("⚠️  Fallback vers create_importance_bar_chart")
                create_importance_bar_chart(
                    cls_importance,
                    output_path=str(output_path),
                    title=title
                )
        else:
            # Pour modèle FTT+ :
            create_importance_bar_chart(
                cls_importance,
                output_path=str(output_path),
                title=title
            )
    def _generate_attention_heatmap(self, model, X: Dict, feature_names: List[str], model_name: str, seed: int):
        """Génère la heatmap d'attention complète."""
        print("Génération de la heatmap d'attention complète...")
        output_path = self.results_dir / 'heatmaps' / f'{model_name}_attention_heatmap_seed_{seed}.png'
        title = f'Heatmap d\'Attention - {model_name} (Seed {seed})'
        
        try:
            from ftt_plus_plus.visualisation import visualize_sparse_attention_heatmap
            visualize_sparse_attention_heatmap(
                model=model,
                x_num=X['test'][0],
                x_cat=X['test'][1],
                feature_names=feature_names,
                output_path=str(output_path),
                title=title
            )
        except ImportError:
            print("⚠️  Fonction visualize_sparse_attention_heatmap non disponible")
    
    def _save_model(self, model_name: str, seed: int, model):
        """Sauvegarde le modèle dans le répertoire results."""
        model_path = self.results_dir / 'best_models' / f'{model_name}_trained_model_weights_seed_{seed}.pt'
        torch.save(model.state_dict(), model_path)
    
    def _save_local_results(
        self,
        output_dir: str,
        training_results: Dict[str, Any],
        performance_results: Dict[str, List[float]],
        cls_importance: Dict[str, float],
        model
    ):
        """Sauvegarde les résultats en local (compatibilité avec l'ancien format)."""
        results_legacy = {
            **training_results,
            'test_performance': performance_results['test'],
            'val_performance': performance_results['val'],
            'cls_importance': cls_importance
        }
        
        np.save(f'{output_dir}/training_results.npy', results_legacy)
        torch.save(model.state_dict(), f'{output_dir}/best_model.pt')
    
    def _print_summary(self, model_name: str, seed: int):
        """Affiche un résumé des fichiers sauvegardés."""
        print(f"\n=== Résultats sauvegardés ===")
        print(f"Métriques de performance: {self.results_dir}/métriques/{model_name}_model_performance_metrics_seed_{seed}.json")
        print(f"Analyse d'importance des features: {self.results_dir}/métriques/{model_name}_feature_importance_analysis_seed_{seed}.json")
        print(f"Graphique d'importance: {self.results_dir}/heatmaps/{model_name}_feature_importance_chart_seed_{seed}.png")
        print(f"Heatmap d'attention: {self.results_dir}/heatmaps/{model_name}_attention_heatmap_seed_{seed}.png")
        print(f"Poids du modèle: {self.results_dir}/best_models/{model_name}_trained_model_weights_seed_{seed}.pt")



def analyze_interpretability(
    model,
    X: Dict,
    y: Dict,
    model_name: str,
    seed: int,
    model_config: Dict[str, Any],
    training_results: Dict[str, Any],
    performance_results: Dict[str, List[float]],
    feature_names: Optional[List[str]] = None,
    local_output_dir: Optional[str] = None,
    results_base_dir: str = 'results/results_telecom'
) -> Dict[str, Any]:
    """
    Fonction de convenance pour l'analyse d'interprétabilité générique.
    
    Interface simplifiée pour utilisation dans tous les scripts d'entraînement FTT.
    
    Args:
        model: modèle FTT entraîné (doit avoir la méthode get_cls_importance)
        X: données d'entrée
        y: labels
        model_name: nom du modèle (ex: 'interpretable_ftt_plus', 'interpretable_ftt_plus_plus')
        seed: seed d'entraînement
        model_config: configuration du modèle
        training_results: résultats d'entraînement
        performance_results: métriques de performance
        feature_names: noms des features (optionnel)
        local_output_dir: répertoire local (optionnel)
        results_base_dir: répertoire de base des résultats
        
    Returns:
        dict: résultats d'interprétabilité
    """
    analyzer = InterpretabilityAnalyzer(results_base_dir)
    return analyzer.analyze_and_save(
        model, X, y, model_name, seed, model_config, training_results, 
        performance_results, feature_names, local_output_dir
    )