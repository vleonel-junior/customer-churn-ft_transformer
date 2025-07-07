"""
Stage1Trainer - Gestionnaire de l'Étape 1 du Pipeline FTT++

Ce module gère l'entraînement du modèle FTT+ et la sélection des features importantes.
Il reçoit les fonctions d'entraînement en paramètre pour éviter les dépendances.
"""

import torch
import torch.nn as nn
import time
from typing import Dict, List, Tuple, Optional, Any, Callable

from ..core.model_ftt_plus import FTTPlusModelWrapper
from ..config.feature_mapping import FeatureMapping


class Stage1Trainer:
    """
    Gestionnaire de l'étape 1 : Entraînement FTT+ et sélection des features importantes.
    
    Cette classe est générique et ne contient aucune dépendance à un dataset spécifique.
    Toutes les fonctions d'entraînement sont passées en paramètre.
    """
    
    def __init__(
        self, 
        feature_mapping: FeatureMapping, 
        ftt_plus_config: Dict[str, Any], 
        M: int, 
        results_dir: str
    ):
        """
        Args:
            feature_mapping: Mapping générique des features
            ftt_plus_config: Configuration du modèle FTT+
            M: Nombre de features à sélectionner
            results_dir: Répertoire de sauvegarde
        """
        self.feature_mapping = feature_mapping
        self.ftt_plus_config = ftt_plus_config
        self.M = M
        self.results_dir = results_dir
    
    def train_ftt_plus(
        self,
        X: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        y: Dict[str, torch.Tensor],
        cat_cardinalities: List[int],
        train_func: Callable,
        val_func: Callable,
        evaluate_func: Callable,
        create_loaders_func: Callable,
        n_epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 64,
        patience: int = 10,
        seed: int = 0,
        embedding_type: str = "LR",
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        Entraîne un modèle FTT+ complet et sélectionne les features importantes.
        
        Args:
            X: Données d'entrée (train, val, test) avec X[split] = (x_num, x_cat)
            y: Labels (train, val, test)
            cat_cardinalities: Cardinalités des features catégorielles
            train_func: Fonction d'entraînement pour une époque
            val_func: Fonction de validation pour une époque
            evaluate_func: Fonction d'évaluation finale
            create_loaders_func: Fonction pour créer les loaders
            n_epochs: Nombre d'époques d'entraînement
            lr: Taux d'apprentissage
            batch_size: Taille des batches
            patience: Patience pour early stopping
            seed: Seed pour la reproductibilité
            embedding_type: Type d'embedding numérique
            device: Device configuré par le script d'entraînement
            
        Returns:
            Résultats de l'étape 1 avec features sélectionnées
        """
        print("🚀 === ÉTAPE 1: Entraînement FTT+ Complet ===")
        print(f"🖥️  Device utilisé: {device}")
        
        # Validation des données avec le mapping
        n_num_features = X['train'][0].shape[1]
        n_cat_features = X['train'][1].shape[1] if X['train'][1] is not None else 0
        self.feature_mapping.validate_data_consistency(n_num_features, n_cat_features)
        
        # Créer et configurer le modèle FTT+
        model_ftt_plus = self._create_ftt_plus_model(
            n_num_features, cat_cardinalities, embedding_type, X, y, device
        )
        
        # Entraînement complet
        training_results = self._train_model(
            model_ftt_plus, X, y, train_func, val_func, create_loaders_func,
            n_epochs, lr, batch_size, patience, device
        )
        
        # Évaluation finale
        performance_results = self._evaluate_model(
            model_ftt_plus, X, y, evaluate_func, seed
        )
        
        # Analyse d'interprétabilité et sélection des features
        interpretability_results, selected_features, feature_importance_scores = (
            self._analyze_and_select_features(
                model_ftt_plus, X, y, seed, n_num_features, cat_cardinalities, 
                embedding_type, training_results, performance_results
            )
        )
        
        # Résultats de l'étape 1
        stage1_results = {
            'model_ftt_plus': model_ftt_plus,
            'selected_features': selected_features,
            'feature_importance_scores': feature_importance_scores,
            'cls_importance_all': interpretability_results['cls_importance'],
            'n_total_features': self.feature_mapping.n_total_features,
            'n_selected_features': self.M,
            'selection_ratio': self.M / self.feature_mapping.n_total_features,
            'training_results': training_results,
            'performance_results': performance_results,
            'interpretability_results': interpretability_results
        }
        
        return stage1_results
    
    def _create_ftt_plus_model(
        self, 
        n_num_features: int, 
        cat_cardinalities: List[int],
        embedding_type: str, 
        X: Dict, 
        y: Dict, 
        device: str
    ) -> FTTPlusModelWrapper:
        """Crée et configure le modèle FTT+."""
        model_ftt_plus = FTTPlusModelWrapper.create_model(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            feature_names=self.feature_mapping.all_feature_names,
            model_config=self.ftt_plus_config,
            device=device
        )
        
        # Embedding numérique personnalisé
        print(f"Type d'embedding numérique: {embedding_type}")
        # Forcer les tenseurs sur CPU pour éviter le warning rtdl_num_embeddings
        X_train_cpu = X['train'][0].cpu()
        model_ftt_plus.configure_num_embedding(
            embedding_type=embedding_type,
            X_train=X_train_cpu,
            d_embedding=self.ftt_plus_config['d_token'],
            y_train=y['train'] if embedding_type in ("T", "T-L", "T-LR", "T-LR-LR") else None
        )
        
        model_info = model_ftt_plus.get_model_info()
        print(f"Modèle FTT+ créé avec {model_info['n_parameters']:,} paramètres")
        
        return model_ftt_plus
    
    def _train_model(
        self, 
        model: FTTPlusModelWrapper, 
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
        """Entraîne le modèle avec early stopping."""
        # Créer les loaders via la fonction passée en paramètre
        train_loader, val_loader = create_loaders_func(y, batch_size, device)
        
        # Optimiseur et fonction de perte
        optimizer = torch.optim.AdamW(model.optimization_param_groups(), lr=lr, weight_decay=0.0)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
        # Entraînement
        print("⏳ Entraînement du modèle FTT+ en cours...")
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
            print(f"✅ Meilleur modèle chargé (époque {best_epoch}, val_loss: {best_val_loss:.4f})")
        
        return {
            'train_losses': train_loss_list,
            'val_losses': val_loss_list,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss
        }
    
    def _evaluate_model(
        self, 
        model: FTTPlusModelWrapper, 
        X: Dict, 
        y: Dict, 
        evaluate_func: Callable,
        seed: int
    ) -> Dict[str, Any]:
        """Évalue le modèle sur validation et test."""
        print("\n📊 Évaluation finale du modèle FTT+...")
        val_performance = evaluate_func(model, 'val', X, y, seed)
        test_performance = evaluate_func(model, 'test', X, y, seed)
        
        return {
            'val': val_performance,
            'test': test_performance
        }
    
    def _analyze_and_select_features(
        self, 
        model: FTTPlusModelWrapper, 
        X: Dict, 
        y: Dict, 
        seed: int,
        n_num_features: int, 
        cat_cardinalities: List[int],
        embedding_type: str, 
        training_results: Dict,
        performance_results: Dict
    ) -> Tuple[Dict, List[str], Dict[str, float]]:
        """Analyse l'interprétabilité et sélectionne les M features importantes."""
        print("\n🔍 Analyse d'interprétabilité avec interpretability_analyzer...")
        
        model_config = {
            'n_num_features': n_num_features,
            'cat_cardinalities': cat_cardinalities,
            'embedding_type': embedding_type,
            **self.ftt_plus_config
        }

        from interpretability_analyzer import analyze_interpretability
        
        # Utiliser interpretability_analyzer pour l'analyse complète
        interpretability_results = analyze_interpretability(
            model=model,
            X=X,
            y=y,
            model_name='interpretable_ftt_plus_from_ftt_plus_plus',
            seed=seed,
            model_config=model_config,
            training_results=training_results,
            performance_results=performance_results,
            feature_names=self.feature_mapping.all_feature_names,
            local_output_dir=None,  # Pas de sauvegarde locale pour l'étape 1
            results_base_dir=str(self.results_dir)
        )
        
        # Extraire l'importance des features depuis les résultats
        cls_importance = interpretability_results['cls_importance']
        
        # Sélectionner les M features les plus importantes
        print(f"\n🎯 Sélection des {self.M} features les plus importantes...")
        sorted_importance = sorted(cls_importance.items(), key=lambda x: x[1], reverse=True)
        selected_features = [name for name, score in sorted_importance[:self.M]]
        feature_importance_scores = {name: score for name, score in sorted_importance[:self.M]}
        
        print("📋 Features sélectionnées:")
        for i, (feature, score) in enumerate(feature_importance_scores.items(), 1):
            feature_type = self.feature_mapping.get_feature_type(feature)
            print(f"  {i:2d}. {feature:<20} ({feature_type}): {score:.4f}")
        
        return interpretability_results, selected_features, feature_importance_scores