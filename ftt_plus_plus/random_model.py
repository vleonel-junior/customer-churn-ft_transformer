"""
Modèle FTT Random pour FTT++ - Étape 2

Ce module implémente le modèle Random utilisé dans la deuxième étape de FTT++.
Il utilise uniquement les M features sélectionnées par l'étape 1 et applique
une attention sparse avec k interactions aléatoires entre features.

Architecture:
-------------
- Feature Tokenizer adapté aux M features sélectionnées
- Token CLS pour l'inférence
- Blocs Transformer avec attention sparse SparseRandomAttention
- Tête de classification

Innovation:
-----------
Ce modèle combine la focalisation de FTT+ (attention CLS-features) avec
une exploration contrôlée des interactions feature-feature, permettant
de capturer des relations importantes tout en maintenant la sparsité.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from torch import Tensor

from .sparse_attention import SparseRandomAttention
from rtdl_lib.modules import FeatureTokenizer, CLSToken, _make_nn_module

ModuleType = str


class SparseTransformerBlock(nn.Module):
    """
    Bloc Transformer avec attention sparse pour le modèle Random de FTT++.
    
    Ce bloc utilise l'attention SparseRandomAttention au lieu de l'attention
    complète, permettant une focalisation sur les interactions essentielles.
    """
    
    def __init__(
        self,
        *,
        d_token: int,
        n_heads: int,
        M: int,  # Nombre de features sélectionnées
        k: int,  # Nombre de paires aléatoires
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: ModuleType,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: ModuleType,
        ffn_normalization: ModuleType,
        residual_dropout: float,
        prenormalization: bool,
        attention_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        
        self.prenormalization = prenormalization
        
        # Mécanisme d'attention sparse
        self.attention = SparseRandomAttention(
            d_model=d_token,
            n_heads=n_heads,
            M=M,
            k=k,
            dropout=attention_dropout,
            initialization=attention_initialization,
            seed=attention_seed
        )
        
        # Normalisation d'attention
        self.attention_normalization = _make_nn_module(attention_normalization, d_token)
        
        # Feed-Forward Network (réutilise l'implémentation RTDL)
        from rtdl_lib.modules import Transformer
        self.ffn = Transformer.FFN(
            d_token=d_token,
            d_hidden=ffn_d_hidden,
            bias_first=True,
            bias_second=True,
            dropout=ffn_dropout,
            activation=ffn_activation
        )
        
        # Normalisation FFN
        self.ffn_normalization = _make_nn_module(ffn_normalization, d_token)
        
        # Dropouts des connexions résiduelles
        self.attention_residual_dropout = nn.Dropout(residual_dropout)
        self.ffn_residual_dropout = nn.Dropout(residual_dropout)
    
    def _start_residual(self, x: Tensor, stage: str) -> Tensor:
        """Démarre une connexion résiduelle avec normalisation pré/post."""
        if self.prenormalization:
            if stage == 'attention':
                return self.attention_normalization(x)
            else:  # stage == 'ffn'
                return self.ffn_normalization(x)
        return x
    
    def _end_residual(self, x: Tensor, x_residual: Tensor, stage: str) -> Tensor:
        """Termine une connexion résiduelle avec dropout et normalisation."""
        if stage == 'attention':
            x_residual = self.attention_residual_dropout(x_residual)
        else:  # stage == 'ffn'
            x_residual = self.ffn_residual_dropout(x_residual)
        
        x = x + x_residual
        
        if not self.prenormalization:
            if stage == 'attention':
                x = self.attention_normalization(x)
            else:  # stage == 'ffn'
                x = self.ffn_normalization(x)
        
        return x
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass avec attention sparse et architecture résiduelle.
        
        Args:
            x: tensor d'entrée de forme (batch_size, 1+M, d_token)
            
        Returns:
            output: tensor transformé de même forme
            attention_weights: poids d'attention sparse moyennés
        """
        # Bloc d'attention sparse avec connexion résiduelle
        x_residual = self._start_residual(x, 'attention')
        x_residual, attention_weights = self.attention(x_residual)
        x = self._end_residual(x, x_residual, 'attention')
        
        # Bloc FFN avec connexion résiduelle  
        x_residual = self._start_residual(x, 'ffn')
        x_residual = self.ffn(x_residual)
        x = self._end_residual(x, x_residual, 'ffn')
        
        return x, attention_weights


class InterpretableFTTRandom(nn.Module):
    """
    Modèle FTT Random pour la deuxième étape de FTT++.
    
    Ce modèle opère uniquement sur les M features les plus importantes
    sélectionnées par un modèle FTT+ pré-entraîné et utilise une attention
    sparse avec k interactions aléatoires entre features.
    
    Args:
        selected_feature_indices_num: Indices des features numériques sélectionnées
        selected_feature_indices_cat: Indices des features catégorielles sélectionnées
        cat_cardinalities_selected: Cardinalités des features catégorielles sélectionnées
        d_token: Taille des tokens
        n_blocks: Nombre de blocs Transformer
        attention_n_heads: Nombre de têtes d'attention
        k: Nombre de paires d'interactions feature-feature aléatoires
        attention_dropout: Taux de dropout de l'attention
        attention_initialization: Initialisation des projections d'attention
        attention_normalization: Type de normalisation de l'attention
        ffn_d_hidden: Taille cachée du feed-forward network
        ffn_dropout: Taux de dropout du FFN
        ffn_activation: Fonction d'activation du FFN
        ffn_normalization: Type de normalisation du FFN
        residual_dropout: Taux de dropout des connexions résiduelles
        prenormalization: Si True, normalisation avant les sous-modules
        head_activation: Fonction d'activation de la tête finale
        head_normalization: Type de normalisation de la tête finale
        d_out: Dimension de sortie
        attention_seed: Seed pour la reproductibilité des interactions aléatoires
    """
    
    def __init__(
        self,
        *,
        selected_feature_indices_num: List[int],
        selected_feature_indices_cat: List[int],
        cat_cardinalities_selected: List[int],
        d_token: int,
        n_blocks: int,
        attention_n_heads: int,
        k: int,  # Nombre de paires d'interactions aléatoires
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: ModuleType,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: ModuleType,
        ffn_normalization: ModuleType,
        residual_dropout: float,
        prenormalization: bool,
        head_activation: ModuleType,
        head_normalization: ModuleType,
        d_out: int,
        attention_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        
        # Sauvegarder les indices des features sélectionnées
        self.selected_feature_indices_num = selected_feature_indices_num
        self.selected_feature_indices_cat = selected_feature_indices_cat
        self.n_num_features_selected = len(selected_feature_indices_num)
        self.n_cat_features_selected = len(selected_feature_indices_cat)
        self.M = self.n_num_features_selected + self.n_cat_features_selected  # Total features sélectionnées
        self.k = k
        
        # Feature Tokenizer adapté aux features sélectionnées
        self.feature_tokenizer = FeatureTokenizer(
            n_num_features=self.n_num_features_selected,
            cat_cardinalities=cat_cardinalities_selected,
            d_token=d_token
        )
        
        # Token CLS pour l'inférence
        self.cls_token = CLSToken(d_token, self.feature_tokenizer.initialization)
        
        # Blocs Transformer avec attention sparse
        self.blocks = nn.ModuleList([
            SparseTransformerBlock(
                d_token=d_token,
                n_heads=attention_n_heads,
                M=self.M,
                k=k,
                attention_dropout=attention_dropout,
                attention_initialization=attention_initialization,
                attention_normalization=attention_normalization,
                ffn_d_hidden=ffn_d_hidden,
                ffn_dropout=ffn_dropout,
                ffn_activation=ffn_activation,
                ffn_normalization=ffn_normalization,
                residual_dropout=residual_dropout,
                prenormalization=prenormalization,
                attention_seed=attention_seed + i if attention_seed is not None else None
            )
            for i in range(n_blocks)
        ])
        
        # Tête de classification
        from rtdl_lib.modules import Transformer
        self.head = Transformer.Head(
            d_in=d_token,
            d_out=d_out,
            bias=True,
            activation=head_activation,
            normalization=head_normalization if prenormalization else 'Identity'
        )
        
        self.prenormalization = prenormalization
    
    @classmethod
    def get_baseline_config(cls) -> Dict[str, Any]:
        """Configuration baseline pour FTT Random."""
        return {
            'attention_n_heads': 8,
            'attention_initialization': 'kaiming',
            'attention_normalization': 'LayerNorm',
            'ffn_activation': 'ReGLU',
            'ffn_normalization': 'LayerNorm',
            'prenormalization': True,
            'head_activation': 'ReLU',
            'head_normalization': 'LayerNorm',
        }
    
    @classmethod
    def from_selected_features(
        cls,
        selected_feature_indices_num: List[int],
        selected_feature_indices_cat: List[int],
        cat_cardinalities_selected: List[int],
        d_token: int,
        n_blocks: int,
        k: int,
        attention_dropout: float,
        ffn_d_hidden: int,
        ffn_dropout: float,
        residual_dropout: float,
        d_out: int,
        attention_seed: Optional[int] = None,
    ) -> 'InterpretableFTTRandom':
        """
        Crée un modèle FTT Random avec configuration baseline.
        
        Args:
            selected_feature_indices_num: Indices des features numériques sélectionnées
            selected_feature_indices_cat: Indices des features catégorielles sélectionnées
            cat_cardinalities_selected: Cardinalités des features catégorielles sélectionnées
            d_token: Taille des tokens
            n_blocks: Nombre de blocs Transformer
            k: Nombre de paires d'interactions aléatoires
            attention_dropout: Taux de dropout de l'attention
            ffn_d_hidden: Taille cachée du FFN
            ffn_dropout: Taux de dropout du FFN
            residual_dropout: Taux de dropout des connexions résiduelles
            d_out: Dimension de sortie
            attention_seed: Seed pour les interactions aléatoires
            
        Returns:
            InterpretableFTTRandom: modèle configuré
        """
        config = cls.get_baseline_config()
        config.update({
            'selected_feature_indices_num': selected_feature_indices_num,
            'selected_feature_indices_cat': selected_feature_indices_cat,
            'cat_cardinalities_selected': cat_cardinalities_selected,
            'd_token': d_token,
            'n_blocks': n_blocks,
            'k': k,
            'attention_dropout': attention_dropout,
            'ffn_d_hidden': ffn_d_hidden,
            'ffn_dropout': ffn_dropout,
            'residual_dropout': residual_dropout,
            'd_out': d_out,
            'attention_seed': attention_seed,
        })
        return cls(**config)
    
    def _select_features(
        self, 
        x_num: Optional[Tensor], 
        x_cat: Optional[Tensor]
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        Sélectionne uniquement les features importantes pour ce modèle.
        
        Args:
            x_num: Features numériques complètes
            x_cat: Features catégorielles complètes
            
        Returns:
            Features numériques et catégorielles sélectionnées
        """
        x_num_selected = None
        x_cat_selected = None
        
        if x_num is not None and self.selected_feature_indices_num:
            x_num_selected = x_num[:, self.selected_feature_indices_num]
        
        if x_cat is not None and self.selected_feature_indices_cat:
            x_cat_selected = x_cat[:, self.selected_feature_indices_cat]
        
        return x_num_selected, x_cat_selected
    
    def forward(
        self, 
        x_num: Optional[Tensor], 
        x_cat: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass du modèle Random avec attention sparse.
        
        Args:
            x_num: Features numériques complètes de forme (batch_size, n_all_num_features)
            x_cat: Features catégorielles complètes de forme (batch_size, n_all_cat_features)
            
        Returns:
            logits: Scores de prédiction de forme (batch_size, d_out)
            last_attention: Poids d'attention de la dernière couche (sparse)
        """
        # Sélectionner uniquement les features importantes
        x_num_selected, x_cat_selected = self._select_features(x_num, x_cat)
        
        # Tokenisation des features sélectionnées
        x = self.feature_tokenizer(x_num_selected, x_cat_selected)
        
        # Ajout du token CLS
        x = self.cls_token(x)
        
        # Vérifier que nous avons bien 1 + M tokens
        expected_seq_len = 1 + self.M
        if x.shape[1] != expected_seq_len:
            raise ValueError(
                f"Séquence de longueur {x.shape[1]} mais {expected_seq_len} attendue "
                f"(1 CLS + {self.M} features sélectionnées)"
            )
        
        # Passage à travers les blocs Transformer avec attention sparse
        last_attention = None
        for block in self.blocks:
            x, attention_weights = block(x)
            last_attention = attention_weights
        
        # Classification à partir du token CLS
        logits = self.head(x)
        
        return logits, last_attention
    
    def get_cls_importance(
        self, 
        x_num: Optional[Tensor], 
        x_cat: Optional[Tensor], 
        selected_feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calcule l'importance des features sélectionnées basée sur l'attention CLS.
        
        Args:
            x_num: Features numériques complètes
            x_cat: Features catégorielles complètes
            selected_feature_names: Noms des features sélectionnées
            
        Returns:
            dict: Mapping feature_name -> importance_score (pour les features sélectionnées uniquement)
        """
        with torch.no_grad():
            _, last_attention = self.forward(x_num, x_cat)
        
        # Moyenner sur le batch et extraire les scores CLS -> features sélectionnées
        avg_attention = last_attention.mean(0)  # (seq_len, seq_len)
        importance_scores = avg_attention[0, 1:].cpu().numpy()  # CLS vers features (exclure CLS->CLS)
        
        if selected_feature_names is not None:
            if len(selected_feature_names) != self.M:
                raise ValueError(
                    f"Nombre de noms ({len(selected_feature_names)}) différent "
                    f"du nombre de features sélectionnées ({self.M})"
                )
            return dict(zip(selected_feature_names, importance_scores))
        else:
            return {f'selected_feature_{i}': score for i, score in enumerate(importance_scores)}
    
    def get_attention_statistics(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur l'attention sparse utilisée.
        
        Returns:
            Statistiques de sparsité et connexions
        """
        # Prendre les statistiques du premier bloc (tous identiques)
        if self.blocks:
            return self.blocks[0].attention.get_attention_statistics()
        return {}
    
    def update_random_interactions(self, new_seed: Optional[int] = None):
        """
        Met à jour les interactions aléatoires dans tous les blocs.
        
        Args:
            new_seed: Nouveau seed de base (sera incrémenté pour chaque bloc)
        """
        for i, block in enumerate(self.blocks):
            block_seed = new_seed + i if new_seed is not None else None
            block.attention.update_random_pairs(block_seed)
    
    def optimization_param_groups(self) -> List[Dict[str, Any]]:
        """Groupes de paramètres optimisés pour l'entraînement (style RTDL)."""
        no_wd_names = ['feature_tokenizer', 'normalization', '.bias']
        
        def needs_wd(name):
            return all(x not in name for x in no_wd_names)
        
        return [
            {'params': [v for k, v in self.named_parameters() if needs_wd(k)]},
            {
                'params': [v for k, v in self.named_parameters() if not needs_wd(k)],
                'weight_decay': 0.0,
            },
        ]