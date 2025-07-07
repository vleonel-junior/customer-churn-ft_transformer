"""
Module d'Analyse d'Interprétabilité Générique

Centralise l'analyse d'interprétabilité pour tous les modèles FTT.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional


class InterpretabilityAnalyzer:
    """Analyseur d'interprétabilité générique pour modèles FTT."""
    
    def __init__(self, results_base_dir: str = 'results'):
        self.results_dir = Path(results_base_dir)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Crée les répertoires nécessaires."""
        for subdir in ['métriques', 'heatmaps', 'best_models']:
            (self.results_dir / subdir).mkdir(parents=True, exist_ok=True)
    
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
        feature_names: List[str],
        local_output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyse complète d'interprétabilité avec sauvegarde."""
        
        if not feature_names:
            raise ValueError("feature_names est obligatoire")
        
        print("\n=== Analyse d'interprétabilité ===")
        
        # Calcul de l'importance
        model.eval()
        with torch.no_grad():
            cls_importance = model.get_cls_importance(X['test'][0], X['test'][1], feature_names)
        
        # Affichage top 10
        sorted_importance = sorted(cls_importance.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 10 features importantes:")
        for i, (feature, score) in enumerate(sorted_importance[:10], 1):
            print(f"  {i:2d}. {feature:<20}: {score:.4f}")
        
        # Sauvegardes
        self._save_metrics(model_name, seed, model_config, training_results, performance_results, model)
        interpretability_results = self._save_interpretability(model_name, seed, feature_names, cls_importance, sorted_importance)
        self._generate_visualizations(model, X, feature_names, model_name, seed, cls_importance)
        self._save_model(model_name, seed, model)
        
        # Suppression de la sauvegarde locale pour éviter les fichiers parasites
        
        self._print_summary(model_name, seed)
        return interpretability_results
    
    def _save_metrics(self, model_name: str, seed: int, model_config: Dict, training_results: Dict, performance_results: Dict, model):
        """Sauvegarde les métriques de performance."""
        data = {
            'model_name': model_name,
            'seed': seed,
            'model_config': model_config,
            'training': {**training_results, 'n_parameters': sum(p.numel() for p in model.parameters())},
            'performance': {
                'val': performance_results['val'],
                'test': performance_results['test']
            }
        }
        
        path = self.results_dir / 'métriques' / f'{model_name}_metrics_seed_{seed}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_interpretability(self, model_name: str, seed: int, feature_names: List[str], cls_importance: Dict, sorted_importance: List) -> Dict:
        """Sauvegarde les résultats d'interprétabilité."""
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
    
    def _generate_visualizations(self, model, X: Dict, feature_names: List[str], model_name: str, seed: int, cls_importance: Dict):
        """Génère les visualisations."""
        print("\nGénération des visualisations...")
        
        # Graphique d'importance
        self._try_visualization(
            [('ftt_plus_plus.visualisation', 'create_ftt_plus_plus_importance_chart'),
             ('ftt_plus.visualisation', 'create_importance_bar_chart')],
            cls_importance,
            output_path=str(self.results_dir / 'heatmaps' / f'{model_name}_importance_seed_{seed}.png'),
            title=f"Importance - {model_name.replace('_', ' ').title()} (Seed {seed})"
        )
        
        # Heatmap d'attention
        # Affichage de la matrice d'attention full désactivé pour ftt_plus
        if "ftt_plus_plus" in model_name:
            self._try_visualization(
                [('ftt_plus_plus.visualisation', 'visualize_sparse_attention_heatmap')],
                model=model,
                x_num=X['test'][0],
                x_cat=X['test'][1],
                feature_names=feature_names,
                output_path=str(self.results_dir / 'heatmaps' / f'{model_name}_attention_seed_{seed}.png'),
                title=f'Attention - {model_name} (Seed {seed})'
            )
        # Pour ftt_plus, on ne génère pas la heatmap d'attention
    
    def _try_visualization(self, functions_to_try: List, *args, **kwargs):
        """Essaie les fonctions de visualisation jusqu'à en trouver une qui marche."""
        called = False
        for module_name, func_name in functions_to_try:
            try:
                module = __import__(module_name, fromlist=[func_name])
                func = getattr(module, func_name)
                print(f"[DEBUG] Tentative d'appel de {module_name}.{func_name}")
                func(*args, **kwargs)
                called = True
                return
            except (ImportError, AttributeError) as e:
                print(f"[DEBUG] Echec import ou appel {module_name}.{func_name} : {e}")
                continue
        if not called:
            raise RuntimeError(
                f"Aucune fonction de visualisation n'a pu être appelée parmi : "
                f"{[f'{m}.{f}' for m, f in functions_to_try]}"
            )
    
    def _save_model(self, model_name: str, seed: int, model):
        """Sauvegarde le modèle."""
        path = self.results_dir / 'best_models' / f'{model_name}_weights_seed_{seed}.pt'
        torch.save(model.state_dict(), path)
    
    def _save_local(self, output_dir: str, training_results: Dict, performance_results: Dict, cls_importance: Dict, model):
        """Sauvegarde locale (compatibilité)."""
        data = {**training_results, 'test_performance': performance_results['test'], 'val_performance': performance_results['val'], 'cls_importance': cls_importance}
        np.save(f'{output_dir}/training_results.npy', data)
        torch.save(model.state_dict(), f'{output_dir}/best_model.pt')
    
    def _print_summary(self, model_name: str, seed: int):
        """Affiche un résumé des fichiers sauvegardés."""
        print(f"\n=== Résultats sauvegardés ===")
        base = f"{self.results_dir}/{model_name}_*_seed_{seed}"
        print(f"Métriques: {base.replace('*', 'metrics')}.json")
        print(f"Importance: {base.replace('*', 'importance')}.json")
        print(f"Visualisations: {base.replace('*', '{importance|attention}')}.png")
        print(f"Modèle: {base.replace('*', 'weights')}.pt")


def analyze_interpretability(
    model,
    X: Dict,
    y: Dict,
    model_name: str,
    seed: int,
    model_config: Dict[str, Any],
    training_results: Dict[str, Any],
    performance_results: Dict[str, List[float]],
    feature_names: List[str],
    local_output_dir: Optional[str] = None,
    results_base_dir: str = 'results'
) -> Dict[str, Any]:
    """Fonction d'analyse d'interprétabilité générique."""
    analyzer = InterpretabilityAnalyzer(results_base_dir)
    return analyzer.analyze_and_save(model, X, y, model_name, seed, model_config, training_results, performance_results, feature_names, local_output_dir)