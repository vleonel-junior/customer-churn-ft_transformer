"""
Stage2Trainer - Gestionnaire de l'Étape 2 du Pipeline FTT++

Ce module gère l'entraînement du modèle Random sur les features sélectionnées.
Il reçoit les fonctions d'entraînement en paramètre pour éviter les dépendances.
"""

import torch
import torch.nn as nn
import time
from typing import Dict, List, Tuple, Optional, Any, Callable

from ..core.model_ftt_random import FTTRandomModel
from ..config.feature_mapping import FeatureMapping


class Stage2Trainer:
    """
    Gestionnaire de l'étape 2 : Entraînement Random sur les features sélectionnées.
    
    Cette classe est générique et ne contient aucune dépendance à un dataset spécifique.
    Toutes les fonctions d'entraînement sont passées en paramètre.
    """
    
    def __init__(
        self, 
        feature_mapping: FeatureMapping, 
        random_model_config: Dict[str, Any],
        k: int, 
        attention_seed: int
    ):
        """
        Args:
            feature_mapping: Mapping générique des features
            random_model_config: Configuration du modèle Random
            k: Nombre d'interactions aléatoires
            attention_seed: Seed pour l'attention
        """
        self.feature_mapping = feature_mapping
        self.random_model_config = random_model_config
        self.k = k
        self.attention_seed = attention_seed
    
    def train_random_model(
        self,
        X: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        y: Dict[str, torch.Tensor],
        cat_cardinalities: List[int],
        selected_features: List[str],
        train_func: Callable,
        val_func: Callable,
        evaluate_func: Callable,
        create_loaders_func: Callable,
        n_epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 64,
        patience: int = 10,
        seed: int = 0,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        Entraîne un modèle Random sur les features sélectionnées.
        
        Args:
            X: Données d'entrée (train, val, test)
            y: Labels (train, val, test)
            cat_cardinalities: Cardinalités des features catégorielles
            selected_features: Features sélectionnées de l'étape 1
            train_func: Fonction d'entraînement pour une époque
            val_func: Fonction de validation pour une époque
            evaluate_func: Fonction d'évaluation finale
            create_loaders_func: Fonction pour créer les loaders
            n_epochs: Nombre d'époques d'entraînement
            lr: Taux d'apprentissage
            batch_size: Taille des batches
            patience: Patience pour early stopping
            seed: Seed pour la reproductibilité
            device: Device configuré par le script d'entraînement
            
        Returns:
            Résultats de l'étape 2
        """
        print("\n🎯 === ÉTAPE 2: Entraînement Modèle Random ===")
        print(f"🖥️  Device utilisé: {device}")
        
        # Créer et configurer le modèle Random
        model_random, attention_stats = self._create_random_model(
            selected_features, cat_cardinalities, device
        )
        
        # Entraînement complet
        training_results = self._train_model(
            model_random, X, y, train_func, val_func, create_loaders_func,
            n_epochs, lr, batch_size, patience, device
        )
        
        # Évaluation finale
        performance_results = self._evaluate_model(
            model_random, X, y, evaluate_func, seed
        )
        
        # Analyse de l'importance dans le modèle Random
        random_importance = self._analyze_random_importance(model_random, X, selected_features)
        
        # Analyse d'interprétabilité complète pour le modèle Random
        self._analyze_random_interpretability(
            model_random, X, y, seed, selected_features, cat_cardinalities,
            training_results, performance_results
        )
        
        # Résultats de l'étape 2
        stage2_results = {
            'model_random': model_random,
            'selected_features': selected_features,
            'selected_indices_num': self.selected_indices_num,
            'selected_indices_cat': self.selected_indices_cat,
            'cat_cardinalities_selected': self.cat_cardinalities_selected,
            'random_importance_scores': random_importance,
            'attention_statistics': attention_stats,
            'sparsity_ratio': attention_stats['sparsity_ratio'],
            'n_parameters': sum(p.numel() for p in model_random.parameters()),
            'training_results': training_results,
            'performance_results': performance_results
        }
        
        return stage2_results
    
    def _create_random_model(
        self, 
        selected_features: List[str], 
        cat_cardinalities: List[int],
        device: str
    ) -> Tuple[FTTRandomModel, Dict[str, Any]]:
        """Crée et configure le modèle Random."""
        # Utiliser le mapping robuste pour déterminer les indices
        selected_indices_num, selected_indices_cat = (
            self.feature_mapping.get_selected_feature_indices(selected_features)
        )
        
        # Sauvegarder pour utilisation ultérieure
        self.selected_indices_num = selected_indices_num
        self.selected_indices_cat = selected_indices_cat
        
        # Cardinalités des features catégorielles sélectionnées
        cat_cardinalities_selected = [cat_cardinalities[i] for i in selected_indices_cat]
        self.cat_cardinalities_selected = cat_cardinalities_selected
        
        print(f"📊 Features sélectionnées: {len(selected_indices_num)} numériques, {len(selected_indices_cat)} catégorielles")
        print(f"🎲 Interactions aléatoires: {self.k} paires")
        
        # Créer le modèle Random
        model_random = FTTRandomModel.create_model(
            selected_feature_indices_num=selected_indices_num,
            selected_feature_indices_cat=selected_indices_cat,
            cat_cardinalities_selected=cat_cardinalities_selected,
            model_config=self.random_model_config,
            k=self.k,
            attention_seed=self.attention_seed
        )
        
        model_random.to(device)
        
        model_info = model_random.get_model_info()
        print(f"Modèle Random créé avec {model_info['n_parameters']:,} paramètres")
        
        # Statistiques d'attention sparse
        attention_stats = model_random.get_attention_statistics()
        print(f"🔗 Sparsité d'attention: {attention_stats['sparsity_ratio']:.2%}")
        print(f"   - Connexions feature-feature: {attention_stats['feature_feature_connections']}")
        
        return model_random, attention_stats
    
    def _train_model(
        self, 
        model: FTTRandomModel, 
        X: Dict, 
        y: Dict, 
        train_func: Callable,
        val_func: Callable,
        create_loaders_func: Callable,
        n_epochs: int,
        lr: float, 
        batch_size: int, 
        patience: int, 
        device: str
    ) -> Dict[str, Any]:
        """Entraîne le modèle Random avec early stopping."""
        # Créer les loaders via la fonction passée en paramètre
        train_loader, val_loader = create_loaders_func(y, batch_size, device)
        
        # Optimiseur et fonction de perte
        optimizer = torch.optim.AdamW(model.optimization_param_groups(), lr=lr, weight_decay=0.0)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
        # Entraînement
        print("⏳ Entraînement du modèle Random en cours...")
        train_loss_list = []
        val_loss_list = []
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(n_epochs):
            start_time = time.time()
            
            # Entraînement via la fonction passée en paramètre
            train_loss = train_func(epoch, model, optimizer, X, y, train_loader, loss_fn)
            
            # Validation via la fonction passée en paramètre
            val_loss = val_func(epoch, model, X, y, val_loader, loss_fn)
            
            # Sauvegarde des métriques
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.2f}s')
            
            # Early stopping et sauvegarde du meilleur modèle
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print(f' <<< NOUVEAU MEILLEUR MODÈLE (val_loss: {val_loss:.4f})')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'\nEarly stopping à l\'époque {epoch} (patience: {patience})')
                    break
        
        # Charger le meilleur modèle
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"✅ Meilleur modèle Random chargé (époque {best_epoch}, val_loss: {best_val_loss:.4f})")
        
        return {
            'train_losses': train_loss_list,
            'val_losses': val_loss_list,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss
        }
    
    def _evaluate_model(
        self, 
        model: FTTRandomModel, 
        X: Dict, 
        y: Dict, 
        evaluate_func: Callable,
        seed: int
    ) -> Dict[str, Any]:
        """Évalue le modèle Random sur validation et test."""
        print("\n📊 Évaluation finale du modèle Random...")
        val_performance = evaluate_func(model, 'val', X, y, seed)
        test_performance = evaluate_func(model, 'test', X, y, seed)
        
        return {
            'val': val_performance,
            'test': test_performance
        }
    
    def _analyze_random_importance(
        self, 
        model: FTTRandomModel, 
        X: Dict,
        selected_features: List[str]
    ) -> Dict[str, float]:
        """Analyse l'importance des features dans le modèle Random."""
        print("\n🔍 Analyse d'importance des features dans le modèle Random...")
        random_importance = model.get_cls_importance(
            X['test'][0], X['test'][1], selected_features
        )
        
        print("\n📊 Importance des features dans le modèle Random:")
        sorted_random_importance = sorted(random_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, score) in enumerate(sorted_random_importance, 1):
            feature_type = self.feature_mapping.get_feature_type(feature)
            print(f"  {i:2d}. {feature:<20} ({feature_type}): {score:.4f}")
        
        return random_importance
    
    def _analyze_random_interpretability(
        self, 
        model: FTTRandomModel, 
        X: Dict, 
        y: Dict,
        seed: int, 
        selected_features: List[str],
        cat_cardinalities: List[int], 
        training_results: Dict,
        performance_results: Dict
    ):
        """Analyse d'interprétabilité complète pour le modèle Random."""
        print("\n🔍 Analyse d'interprétabilité complète du modèle Random...")
        
        model_config = {
            'selected_features': selected_features,
            'k': self.k,
            'attention_seed': self.attention_seed,
            'selected_indices_num': self.selected_indices_num,
            'selected_indices_cat': self.selected_indices_cat,
            'cat_cardinalities_selected': self.cat_cardinalities_selected,
            **self.random_model_config
        }
        
        from interpretability_analyzer import analyze_interpretability
        
        # Utiliser interpretability_analyzer pour l'analyse complète du modèle Random
        analyze_interpretability(
            model=model,
            X=X,
            y=y,
            model_name='interpretable_ftt_plus_plus',
            seed=seed,
            model_config=model_config,
            training_results=training_results,
            performance_results=performance_results,
            feature_names=selected_features,  # Seulement les features sélectionnées
            local_output_dir=None,
            results_base_dir='results/results_telecom'
        )