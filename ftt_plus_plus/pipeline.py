"""
Pipeline FTT++ - Orchestration Complète des Deux Étapes

Ce module implémente le pipeline complet FTT++ qui combine :
1. Entraînement d'un modèle FTT+ et sélection des M features importantes
2. Entraînement d'un modèle Random sur les features sélectionnées

Le pipeline gère automatiquement :
- L'entraînement complet du modèle FTT+ (étape 1)
- L'analyse d'interprétabilité via interpretability_analyzer
- La sélection des M features les plus importantes
- L'entraînement du modèle Random sur les features sélectionnées (étape 2)
- L'évaluation comparative des performances

Usage:
------
pipeline = FTTPlusPlusPipeline(M=10, k=5)
results = pipeline.run_complete_pipeline(X, y, cat_cardinalities)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import time

from .config import FTTPlusPlusConfig, FeatureMapping
from .training_stages import Stage1Trainer, Stage2Trainer


class FTTPlusPlusPipeline:
    """
    Pipeline complet pour FTT++ - Approche en deux étapes avec entraînement intégré.
    
    Cette classe orchestre l'ensemble du processus FTT++ :
    1. Entraînement complet FTT+ → Analyse d'interprétabilité → Sélection des M features
    2. Entraînement Random sur les features sélectionnées → Modèle final optimisé
    """
    
    def __init__(
        self,
        config: Optional[FTTPlusPlusConfig] = None,
        M: int = 10,
        k: int = 5,
        attention_seed: int = 42,
        results_dir: str = 'results/results_telecom',
        feature_mapping: Optional[FeatureMapping] = None
    ):
        """
        Args:
            config: Configuration complète (si None, utilise les autres paramètres)
            M: Nombre de features à sélectionner
            k: Nombre d'interactions aléatoires
            attention_seed: Seed pour la reproductibilité
            results_dir: Répertoire de sauvegarde
            feature_mapping: Mapping explicite des features (si None, utilise Telecom par défaut)
        """
        if config is not None:
            self.config = config
        else:
            # Configuration par défaut
            self.config = FTTPlusPlusConfig(
                ftt_plus_config=self._get_default_ftt_plus_config(),
                M=M,
                k=k,
                random_model_config=self._get_default_random_config(),
                attention_seed=attention_seed,
                results_dir=results_dir
            )
        
        # Mapping des features
        if feature_mapping is None:
            self.feature_mapping = FeatureMapping.from_telecom_dataset()
            print("Utilisation du mapping Telecom par défaut")
        else:
            self.feature_mapping = feature_mapping
            print("Utilisation du mapping personnalisé")

        self.results_dir = Path(self.config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Créer les gestionnaires d'étapes
        self.stage1_trainer = Stage1Trainer(
            feature_mapping=self.feature_mapping,
            ftt_plus_config=self.config.ftt_plus_config,
            M=self.config.M,
            results_dir=str(self.results_dir)
        )
        
        self.stage2_trainer = Stage2Trainer(
            feature_mapping=self.feature_mapping,
            random_model_config=self.config.random_model_config,
            k=self.config.k,
            attention_seed=self.config.attention_seed
        )
        
        # Résultats intermédiaires
        self.stage1_results = None
        self.stage2_results = None
        self.selected_features = None
        self.feature_importance_scores = None
    
    def _get_default_ftt_plus_config(self) -> Dict[str, Any]:
        """Configuration par défaut pour FTT+."""
        return {
            'd_token': 64,
            'n_blocks': 3,
            'attention_dropout': 0.1,
            'ffn_d_hidden': 128,
            'ffn_dropout': 0.1,
            'residual_dropout': 0.1,
            'd_out': 1
        }
    
    def _get_default_random_config(self) -> Dict[str, Any]:
        """Configuration par défaut pour le modèle Random."""
        return {
            'd_token': 64,
            'n_blocks': 3,
            'attention_dropout': 0.1,
            'ffn_d_hidden': 128,
            'ffn_dropout': 0.1,
            'residual_dropout': 0.1,
            'd_out': 1
        }
    
    def stage1_train_ftt_plus(
        self,
        X: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        y: Dict[str, torch.Tensor],
        cat_cardinalities: List[int],
        n_epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 64,
        patience: int = 10,
        seed: int = 0,
        embedding_type: str = "LR",
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        Étape 1 : Entraîne un modèle FTT+ et sélectionne les features importantes.
        
        Cette méthode délègue à Stage1Trainer pour gérer l'entraînement et la sélection.
        
        Args:
            device: Device configuré par le script d'entraînement
        """
        stage1_results = self.stage1_trainer.train_ftt_plus(
            X=X,
            y=y,
            cat_cardinalities=cat_cardinalities,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            patience=patience,
            seed=seed,
            embedding_type=embedding_type,
            device=device
        )
        
        # Sauvegarder les résultats
        self.selected_features = stage1_results['selected_features']
        self.feature_importance_scores = stage1_results['feature_importance_scores']
        
        # Sauvegarder si demandé
        if self.config.save_intermediate:
            self._save_stage1_results(stage1_results)
        
        self.stage1_results = stage1_results
        return stage1_results
    
    def stage2_train_random_model(
        self,
        X: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        y: Dict[str, torch.Tensor],
        cat_cardinalities: List[int],
        stage1_results: Optional[Dict[str, Any]] = None,
        n_epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 64,
        patience: int = 10,
        seed: int = 0,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        Étape 2 : Entraîne un modèle Random sur les features sélectionnées.
        
        Cette méthode délègue à Stage2Trainer pour gérer l'entraînement du modèle Random.
        
        Args:
            device: Device configuré par le script d'entraînement 
        """
        if stage1_results is None:
            if self.stage1_results is None:
                raise ValueError("Aucun résultat de l'étape 1 disponible. Exécutez d'abord stage1_train_ftt_plus()")
            stage1_results = self.stage1_results
        
        selected_features = stage1_results['selected_features']
        
        stage2_results = self.stage2_trainer.train_random_model(
            X=X,
            y=y,
            cat_cardinalities=cat_cardinalities,
            selected_features=selected_features,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            patience=patience,
            seed=seed,
            device=device
        )
        
        # Sauvegarder si demandé
        if self.config.save_intermediate:
            self._save_stage2_results(stage2_results)
        
        self.stage2_results = stage2_results
        return stage2_results
    
    def run_complete_pipeline(
        self,
        X: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        y: Dict[str, torch.Tensor],
        cat_cardinalities: List[int],
        stage1_epochs: int = 50,
        stage2_epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 64,
        patience: int = 10,
        seed: int = 0,
        embedding_type: str = "LR",
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        Exécute le pipeline complet FTT++ (Étape 1 + Étape 2).
        
        Args:
            device: Device déjà configuré par setup_device()
        
        Returns:
            Résultats complets du pipeline avec comparaisons
        """
        print("🚀 === PIPELINE FTT++ COMPLET ===")
        print(f"🖥️  Device utilisé: {device}")
        start_time = time.time()
        
        # Étape 1: FTT+ avec entraînement complet
        stage1_results = self.stage1_train_ftt_plus(
            X, y, cat_cardinalities,
            n_epochs=stage1_epochs, lr=lr, batch_size=batch_size, 
            patience=patience, seed=seed, embedding_type=embedding_type, device=device
        )
        
        # Étape 2: Random sur features sélectionnées
        stage2_results = self.stage2_train_random_model(
            X, y, cat_cardinalities, stage1_results,
            n_epochs=stage2_epochs, lr=lr, batch_size=batch_size,
            patience=patience, seed=seed, device=device
        )
        
        # Analyse comparative
        comparison_results = self._compare_stages(stage1_results, stage2_results)
        
        total_time = time.time() - start_time
        
        # Résultats finaux
        complete_results = {
            'pipeline_config': self.config,
            'stage1_results': stage1_results,
            'stage2_results': stage2_results,
            'comparison': comparison_results,
            'execution_time': total_time,
            'selected_features': self.selected_features,
            'feature_mapping': {
                'num_features': self.feature_mapping.num_feature_names,
                'cat_features': self.feature_mapping.cat_feature_names,
                'all_features': self.feature_mapping.all_feature_names
            },
            'feature_importance_evolution': {
                'ftt_plus': self.feature_importance_scores,
                'random': stage2_results['random_importance_scores']
            }
        }
        
        # Sauvegarde finale
        self._save_complete_results(complete_results)
        
        print(f"\n✅ === PIPELINE TERMINÉ EN {total_time:.1f}s ===")
        return complete_results
    
    def _compare_stages(self, stage1_results: Dict, stage2_results: Dict) -> Dict[str, Any]:
        """Compare les résultats des deux étapes."""
        
        # Comparaison des modèles
        ftt_plus_params = sum(p.numel() for p in stage1_results['model_ftt_plus'].parameters())
        random_params = sum(p.numel() for p in stage2_results['model_random'].parameters())
        
        # Comparaison de l'importance des features
        ftt_plus_importance = stage1_results['feature_importance_scores']
        random_importance = stage2_results['random_importance_scores']
        
        # Corrélation entre les importances
        ftt_plus_scores = [ftt_plus_importance[f] for f in self.selected_features]
        random_scores = [random_importance[f] for f in self.selected_features]
        correlation = np.corrcoef(ftt_plus_scores, random_scores)[0, 1]
        
        return {
            'parameter_reduction': {
                'ftt_plus': ftt_plus_params,
                'random': random_params,
                'reduction_ratio': (ftt_plus_params - random_params) / ftt_plus_params,
                'reduction_absolute': ftt_plus_params - random_params
            },
            'feature_importance_correlation': correlation,
            'sparsity_achieved': stage2_results['sparsity_ratio'],
            'features_retained': len(self.selected_features),
            'feature_retention_ratio': len(self.selected_features) / stage1_results['n_total_features']
        }
    
    def _save_stage1_results(self, results: Dict):
        """Sauvegarde les résultats de l'étape 1."""
        save_path = self.results_dir / 'ftt_plus_plus_stage1_results.json'
        
        save_data = {
            'selected_features': results['selected_features'],
            'feature_importance_scores': results['feature_importance_scores'],
            'n_total_features': results['n_total_features'],
            'n_selected_features': results['n_selected_features'],
            'selection_ratio': results['selection_ratio'],
            'feature_mapping': {
                'num_features': self.feature_mapping.num_feature_names,
                'cat_features': self.feature_mapping.cat_feature_names,
                'all_features': self.feature_mapping.all_feature_names
            }
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Résultats étape 1 sauvegardés: {save_path}")
    
    def _save_stage2_results(self, results: Dict):
        """Sauvegarde les résultats de l'étape 2."""
        save_path = self.results_dir / 'ftt_plus_plus_stage2_results.json'
        
        save_data = {
            'selected_features': results['selected_features'],
            'random_importance_scores': results['random_importance_scores'],
            'attention_statistics': results['attention_statistics'],
            'sparsity_ratio': results['sparsity_ratio'],
            'n_parameters': results['n_parameters']
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Résultats étape 2 sauvegardés: {save_path}")
    
    def _save_complete_results(self, results: Dict):
        """Sauvegarde les résultats complets du pipeline."""
        save_path = self.results_dir / 'ftt_plus_plus_results.json'
        
        # Préparer les données à sauvegarder (exclure les modèles)
        save_data = {
            'pipeline_config': {
                'M': self.config.M,
                'k': self.config.k,
                'attention_seed': self.config.attention_seed
            },
            'execution_time': results['execution_time'],
            'selected_features': results['selected_features'],
            'feature_mapping': results['feature_mapping'],
            'feature_importance_evolution': results['feature_importance_evolution'],
            'comparison': results['comparison']
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Résultats complets sauvegardés: {save_path}")