"""
Étapes d'Entraînement pour FTT++

Ce module contient les deux étapes d'entraînement du pipeline FTT++ :
1. Stage1Trainer - Entraînement FTT+ et sélection des features importantes
2. Stage2Trainer - Entraînement Random sur les features sélectionnées
"""

import torch
import torch.nn as nn
import time
from typing import Dict, List, Tuple, Optional, Any

from .random_model import InterpretableFTTRandom
from .config import FeatureMapping
from ftt_plus.model import InterpretableFTTPlus

from train.Telecom.train_ftt_plus.train_func import train, val, evaluate
from num_embedding_factory import get_num_embedding
import zero


class Stage1Trainer:
    """
    Gestionnaire de l'étape 1 : Entraînement FTT+ et sélection des features importantes.
    """
    
    def __init__(self, feature_mapping: FeatureMapping, ftt_plus_config: Dict[str, Any], 
                 M: int, results_dir: str):
        """
        Args:
            feature_mapping: Mapping des features
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
            n_epochs: Nombre d'époques d'entraînement
            lr: Taux d'apprentissage
            batch_size: Taille des batches
            patience: Patience pour early stopping
            seed: Seed pour la reproductibilité
            embedding_type: Type d'embedding numérique
            device: Device configuré par setup_device() dans le script d'entraînement
            
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
            model_ftt_plus, X, y, n_epochs, lr, batch_size, patience, device
        )
        
        # Évaluation finale
        performance_results = self._evaluate_model(model_ftt_plus, X, y, seed)
        
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
            'n_total_features': len(self.feature_mapping.all_feature_names),
            'n_selected_features': self.M,
            'selection_ratio': self.M / len(self.feature_mapping.all_feature_names),
            'training_results': training_results,
            'performance_results': performance_results,
            'interpretability_results': interpretability_results
        }
        
        return stage1_results
    
    def _create_ftt_plus_model(self, n_num_features: int, cat_cardinalities: List[int],
                              embedding_type: str, X: Dict, y: Dict, device: str) -> InterpretableFTTPlus:
        """Crée et configure le modèle FTT+."""
        model_ftt_plus = InterpretableFTTPlus.make_baseline(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            **self.ftt_plus_config
        )
        
        # Embedding numérique personnalisé
        print(f"Type d'embedding numérique: {embedding_type}")
        num_embedding = get_num_embedding(
            embedding_type=embedding_type,
            X_train=X['train'][0],
            d_embedding=self.ftt_plus_config['d_token'],
            y_train=y['train'] if embedding_type in ("T", "T-L", "T-LR", "T-LR-LR") else None
        )
        model_ftt_plus.feature_tokenizer.num_tokenizer = num_embedding
        
        model_ftt_plus.to(device)
        
        print(f"Modèle FTT+ créé avec {sum(p.numel() for p in model_ftt_plus.parameters()):,} paramètres")
        
        return model_ftt_plus
    
    def _train_model(self, model: nn.Module, X: Dict, y: Dict, n_epochs: int,
                    lr: float, batch_size: int, patience: int, device: str) -> Dict[str, Any]:
        """Entraîne le modèle avec early stopping."""
        # Créer les loaders
        train_loader = zero.data.IndexLoader(len(y['train']), batch_size, device=device)
        val_loader = zero.data.IndexLoader(len(y['val']), batch_size, device=device)
        
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
            
            # Entraînement
            train_loss = train(epoch, model, optimizer, X, y, train_loader, loss_fn)
            
            # Validation
            val_loss = val(epoch, model, X, y, val_loader, loss_fn)
            
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
    
    def _evaluate_model(self, model: nn.Module, X: Dict, y: Dict, seed: int) -> Dict[str, Any]:
        """Évalue le modèle sur validation et test."""
        print("\n Évaluation finale du modèle FTT+...")
        val_performance = evaluate(model, 'val', X, y, seed)
        test_performance = evaluate(model, 'test', X, y, seed)
        
        return {
            'val': val_performance,
            'test': test_performance
        }
    
    def _analyze_and_select_features(self, model: nn.Module, X: Dict, y: Dict, seed: int,
                                   n_num_features: int, cat_cardinalities: List[int],
                                   embedding_type: str, training_results: Dict,
                                   performance_results: Dict) -> Tuple[Dict, List[str], Dict[str, float]]:
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
            model_name='interpretable_ftt_plus',
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
            feature_type = "NUM" if feature in self.feature_mapping.num_feature_names else "CAT"
            print(f"  {i:2d}. {feature:<20} ({feature_type}): {score:.4f}")
        
        return interpretability_results, selected_features, feature_importance_scores


class Stage2Trainer:
    """
    Gestionnaire de l'étape 2 : Entraînement Random sur les features sélectionnées.
    """
    
    def __init__(self, feature_mapping: FeatureMapping, random_model_config: Dict[str, Any],
                 k: int, attention_seed: int):
        """
        Args:
            feature_mapping: Mapping des features
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
            n_epochs: Nombre d'époques d'entraînement
            lr: Taux d'apprentissage
            batch_size: Taille des batches
            patience: Patience pour early stopping
            seed: Seed pour la reproductibilité
            device: Device configuré par setup_device() dans le script d'entraînement
            
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
            model_random, X, y, n_epochs, lr, batch_size, patience, device
        )
        
        # Évaluation finale
        performance_results = self._evaluate_model(model_random, X, y, seed)
        
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
    
    def _create_random_model(self, selected_features: List[str], cat_cardinalities: List[int],
                           device: str) -> Tuple[InterpretableFTTRandom, Dict[str, Any]]:
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
        model_random = InterpretableFTTRandom.from_selected_features(
            selected_feature_indices_num=selected_indices_num,
            selected_feature_indices_cat=selected_indices_cat,
            cat_cardinalities_selected=cat_cardinalities_selected,
            k=self.k,
            attention_seed=self.attention_seed,
            **self.random_model_config
        )
        
        model_random.to(device)
        
        print(f"Modèle Random créé avec {sum(p.numel() for p in model_random.parameters()):,} paramètres")
        
        # Statistiques d'attention sparse
        attention_stats = model_random.get_attention_statistics()
        print(f"🔗 Sparsité d'attention: {attention_stats['sparsity_ratio']:.2%}")
        print(f"   - Connexions feature-feature: {attention_stats['feature_feature_connections']}")
        
        return model_random, attention_stats
    
    def _train_model(self, model: nn.Module, X: Dict, y: Dict, n_epochs: int,
                    lr: float, batch_size: int, patience: int, device: str) -> Dict[str, Any]:
        """Entraîne le modèle Random avec early stopping."""
        # Créer les loaders
        train_loader = zero.data.IndexLoader(len(y['train']), batch_size, device=device)
        val_loader = zero.data.IndexLoader(len(y['val']), batch_size, device=device)
        
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
            
            # Entraînement
            train_loss = train(epoch, model, optimizer, X, y, train_loader, loss_fn)
            
            # Validation
            val_loss = val(epoch, model, X, y, val_loader, loss_fn)
            
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
    
    def _evaluate_model(self, model: nn.Module, X: Dict, y: Dict, seed: int) -> Dict[str, Any]:
        """Évalue le modèle Random sur validation et test."""
        print("\n Évaluation finale du modèle Random...")
        val_performance = evaluate(model, 'val', X, y, seed)
        test_performance = evaluate(model, 'test', X, y, seed)
        
        return {
            'val': val_performance,
            'test': test_performance
        }
    
    def _analyze_random_importance(self, model: InterpretableFTTRandom, X: Dict,
                                 selected_features: List[str]) -> Dict[str, float]:
        """Analyse l'importance des features dans le modèle Random."""
        print("\n🔍 Analyse d'importance des features dans le modèle Random...")
        random_importance = model.get_cls_importance(
            X['test'][0], X['test'][1], selected_features
        )
        
        print("\n Importance des features dans le modèle Random:")
        sorted_random_importance = sorted(random_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, score) in enumerate(sorted_random_importance, 1):
            feature_type = "NUM" if feature in self.feature_mapping.num_feature_names else "CAT"
            print(f"  {i:2d}. {feature:<20} ({feature_type}): {score:.4f}")
        
        return random_importance
    
    def _analyze_random_interpretability(self, model: InterpretableFTTRandom, X: Dict, y: Dict,
                                       seed: int, selected_features: List[str],
                                       cat_cardinalities: List[int], training_results: Dict,
                                       performance_results: Dict):
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