"""
FTT+ Interprétable - Modèle optimisé inspiré de RTDL avec Sparsemax

Cette implémentation combine:
1. L'architecture robuste de RTDL
2. L'attention sélective FTT+ d'Isomura et al.
3. L'interprétabilité multi-têtes inspirée du TFT
4. L'utilisation de sparsemax pour une attention plus creuse et interprétable

Références:
    * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    * [isomura2023optimizing] Tokimasa Isomura, Ryotaro Shimizu, Masayuki Goto, "Optimizing FT-Transformer: Sparse Attention for Improved Performance and Interpretability", 2023
"""

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from torch import Tensor

from .attention import InterpretableMultiHeadAttention
from rtdl_lib.modules import FeatureTokenizer, CLSToken, _make_nn_module

ModuleType = Union[str, Callable[..., nn.Module]]


class InterpretableTransformerBlock(nn.Module):
    """Bloc Transformer avec attention interprétable et architecture RTDL.
    
    Cette implémentation suit l'architecture des blocs Transformer de RTDL
    tout en intégrant le mécanisme d'attention interprétable FTT+ avec sparsemax.
    
    Args:
        d_token: la taille des tokens d'entrée et de sortie
        n_heads: le nombre de têtes d'attention
        attention_dropout: taux de dropout pour l'attention
        attention_initialization: politique d'initialisation pour l'attention
        attention_normalization: type de normalisation pour l'attention
        ffn_d_hidden: taille cachée du réseau feed-forward
        ffn_dropout: taux de dropout du FFN
        ffn_activation: fonction d'activation du FFN
        ffn_normalization: type de normalisation du FFN
        residual_dropout: taux de dropout des connexions résiduelles
        prenormalization: si True, applique la normalisation avant les sous-modules
        
    Example:
        .. testcode::
        
            block = InterpretableTransformerBlock(
                d_token=128,
                n_heads=8,
                attention_dropout=0.1,
                attention_initialization='kaiming',
                attention_normalization='LayerNorm',
                ffn_d_hidden=256,
                ffn_dropout=0.1,
                ffn_activation='ReGLU',
                ffn_normalization='LayerNorm',
                residual_dropout=0.0,
                prenormalization=True
            )
            x = torch.randn(4, 10, 128)
            output, attention_weights = block(x)
            assert output.shape == x.shape
    """
    
    def __init__(
        self,
        *,
        d_token: int,
        n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: ModuleType,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: ModuleType,
        ffn_normalization: ModuleType,
        residual_dropout: float,
        prenormalization: bool,
        attention_mode: str = 'hybrid',  # Défaut = 'hybrid'
    ) -> None:
        super().__init__()
        
        self.prenormalization = prenormalization
        
        # Mécanisme d'attention interprétable avec sparsemax
        self.attention = InterpretableMultiHeadAttention(
            d_model=d_token,
            n_heads=n_heads,
            dropout=attention_dropout,
            initialization=attention_initialization,
            attention_mode=attention_mode  # Passer le paramètre
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
        """Démarre une connexion résiduelle avec normalisation pré/post selon la configuration."""
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
        """Forward pass avec attention interprétable et architecture résiduelle.
        
        Args:
            x: tensor d'entrée de forme (batch_size, seq_len, d_token)
            
        Returns:
            output: tensor transformé de même forme que l'entrée
            attention_weights: poids d'attention moyennés de forme (batch_size, seq_len, seq_len)
        """
        # Bloc d'attention avec connexion résiduelle
        x_residual = self._start_residual(x, 'attention')
        x_residual, attention_weights = self.attention(x_residual)
        x = self._end_residual(x, x_residual, 'attention')
        
        # Bloc FFN avec connexion résiduelle  
        x_residual = self._start_residual(x, 'ffn')
        x_residual = self.ffn(x_residual)
        x = self._end_residual(x, x_residual, 'ffn')
        
        return x, attention_weights


class InterpretableFTTPlus(nn.Module):
    """FT-Transformer interprétable avec attention sélective FTT+ et sparsemax.
    
    Cette implémentation combine l'architecture robuste du FT-Transformer RTDL
    avec les innovations d'attention interprétable pour données tabulaires et sparsemax.
    
    Caractéristiques principales:
        * Feature Tokenizer RTDL pour l'embedding optimal
        * Token CLS pour l'inférence BERT-like  
        * Blocs Transformer avec attention interprétable et sparsemax
        * Méthodes d'explicabilité intégrées
        
    Args:
        n_num_features: nombre de features numériques continues
        cat_cardinalities: cardinalités des features catégorielles
        d_token: taille des tokens (doit être divisible par n_heads)
        n_blocks: nombre de blocs Transformer
        n_heads: nombre de têtes d'attention
        attention_dropout: taux de dropout de l'attention
        attention_initialization: initialisation des projections d'attention
        attention_normalization: type de normalisation de l'attention
        ffn_d_hidden: taille cachée du feed-forward network
        ffn_dropout: taux de dropout du FFN
        ffn_activation: fonction d'activation du FFN  
        ffn_normalization: type de normalisation du FFN
        residual_dropout: taux de dropout des connexions résiduelles
        prenormalization: si True, normalisation avant les sous-modules
        head_activation: fonction d'activation de la tête finale
        head_normalization: type de normalisation de la tête finale
        d_out: dimension de sortie
        
    Example:
        .. testcode::
        
            model = InterpretableFTTPlus.make_baseline(
                n_num_features=3,
                cat_cardinalities=[2, 5, 10],
                d_token=128,
                n_blocks=3,
                attention_dropout=0.1,
                ffn_d_hidden=256,
                ffn_dropout=0.1,
                residual_dropout=0.0,
                d_out=1
            )
            
            x_num = torch.randn(32, 3)
            x_cat = torch.randint(0, 5, (32, 3))
            
            logits, attention = model(x_num, x_cat)
            assert logits.shape == (32, 1)
            
            # Analyse d'interprétabilité
            importance = model.get_cls_importance(x_num[:1], x_cat[:1])
    """
    
    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: List[int],
        d_token: int,
        n_blocks: int,
        n_heads: int,
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
        attention_mode: str = 'hybrid',
    ) -> None:
        super().__init__()
        
        # Feature Tokenizer RTDL
        self.feature_tokenizer = FeatureTokenizer(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_token=d_token
        )
        
        # Token CLS pour l'inférence BERT-like
        self.cls_token = CLSToken(d_token, self.feature_tokenizer.initialization)
        
        # Blocs Transformer interprétables avec sparsemax
        self.blocks = nn.ModuleList([
            InterpretableTransformerBlock(
                d_token=d_token,
                n_heads=n_heads,
                attention_dropout=attention_dropout,
                attention_initialization=attention_initialization,
                attention_normalization=attention_normalization,
                ffn_d_hidden=ffn_d_hidden,
                ffn_dropout=ffn_dropout,
                ffn_activation=ffn_activation,
                ffn_normalization=ffn_normalization,
                residual_dropout=residual_dropout,
                prenormalization=prenormalization,
                attention_mode=attention_mode
            )
            for _ in range(n_blocks)
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
        """Configuration baseline optimisée pour FTT+ interprétable.
        
        Cette configuration suit les meilleures pratiques RTDL tout en
        optimisant pour l'interprétabilité des données tabulaires.
        
        Returns:
            dict: configuration des hyperparamètres baseline
        """
        return {
            'n_heads': 8,
            'attention_initialization': 'kaiming',
            'attention_normalization': 'LayerNorm',
            'ffn_activation': 'ReGLU',
            'ffn_normalization': 'LayerNorm',
            'prenormalization': True,
            'head_activation': 'ReLU',
            'head_normalization': 'LayerNorm',
            'attention_mode': 'hybrid',
        }
    
    @classmethod
    def make_baseline(
        cls,
        *,
        n_num_features: int,
        cat_cardinalities: List[int],
        d_token: int,
        n_blocks: int,
        n_heads: int,
        attention_dropout: float,
        ffn_d_hidden: int,
        ffn_dropout: float,
        residual_dropout: float,
        d_out: int,
        attention_mode: str = 'hybrid',
    ) -> 'InterpretableFTTPlus':
        """Crée un modèle FTT+ interprétable avec configuration baseline.
        
        Cette méthode est le constructeur recommandé qui combine les
        meilleures pratiques RTDL avec les optimisations FTT+.
        
        Args:
            n_num_features: nombre de features numériques continues
            cat_cardinalities: liste des cardinalités des features catégorielles
            d_token: taille des tokens (doit être divisible par 8)
            n_blocks: nombre de blocs Transformer
            attention_dropout: taux de dropout de l'attention (>0 recommandé)
            ffn_d_hidden: taille cachée du FFN (recommandé: 2-4x d_token)
            ffn_dropout: taux de dropout du FFN
            residual_dropout: taux de dropout des connexions résiduelles
            d_out: dimension de sortie (1 pour classification binaire)
            
        Returns:
            InterpretableFTTPlus: modèle configuré avec les paramètres baseline
            
        Example:
            .. testcode::
            
                model = InterpretableFTTPlus.make_baseline(
                    n_num_features=5,
                    cat_cardinalities=[3, 4, 10],
                    d_token=128,
                    n_blocks=3,
                    attention_dropout=0.1,
                    ffn_d_hidden=256,
                    ffn_dropout=0.1,
                    residual_dropout=0.0,
                    d_out=1
                )
        """
        config = cls.get_baseline_config()
        config.update({
            'n_num_features': n_num_features,
            'cat_cardinalities': cat_cardinalities,
            'd_token': d_token,
            'n_blocks': n_blocks,
            'n_heads': n_heads,  #
            'attention_dropout': attention_dropout,
            'ffn_d_hidden': ffn_d_hidden,
            'ffn_dropout': ffn_dropout,
            'residual_dropout': residual_dropout,
            'd_out': d_out,
            'attention_mode': attention_mode,
        })
        return cls(**config)
    
    def forward(self, x_num: Optional[Tensor], x_cat: Optional[Tensor], return_attention: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass du modèle avec extraction d'attention optionnelle.
        
        Args:
            x_num: features numériques de forme (batch_size, n_num_features)
            x_cat: features catégorielles de forme (batch_size, n_cat_features)
            return_attention: si True, retourne aussi les poids d'attention
        
        Returns:
            logits: scores de prédiction de forme (batch_size, d_out)
            last_attention: poids d'attention de la dernière couche (batch_size, seq_len, seq_len) si demandé
        """
        # Tokenisation des features
        x = self.feature_tokenizer(x_num, x_cat)
        
        # Ajout du token CLS
        x = self.cls_token(x)
        
        # Passage à travers les blocs Transformer
        last_attention = None
        for block in self.blocks:
            x, attention_weights = block(x)
            last_attention = attention_weights
        
        # Classification à partir du token CLS
        logits = self.head(x)
        
        if return_attention:
            return logits, last_attention
        else:
            return logits
    
    def get_cls_importance(
        self, 
        x_num: Optional[Tensor], 
        x_cat: Optional[Tensor], 
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Calcule l'importance des features basée sur l'attention CLS.
        
        Cette méthode extrait les scores d'attention entre le token CLS et
        chaque feature, fournissant une mesure directe d'importance.
        
        Args:
            x_num: features numériques (batch unique ou batch)
            x_cat: features catégorielles (batch unique ou batch)  
            feature_names: noms des features pour labelliser les résultats
            
        Returns:
            dict: mapping feature_name -> importance_score
            
        Example:
            .. testcode::
            
                importance = model.get_cls_importance(x_num, x_cat, 
                    ['feature_1', 'feature_2', 'feature_3'])
                print(f"Feature la plus importante: {max(importance, key=importance.get)}")
        """
        with torch.no_grad():
            _, last_attention = self.forward(x_num, x_cat, return_attention=True)
        
        # Moyenner sur le batch et extraire les scores CLS -> features
        avg_attention = last_attention.mean(0)  # (seq_len, seq_len)
        importance_scores = avg_attention[0, 1:].cpu().numpy()  # Exclure CLS->CLS
        
        if feature_names is not None:
            return dict(zip(feature_names, importance_scores))
        else:
            total_features = len(importance_scores)
            return {f'feature_{i}': score for i, score in enumerate(importance_scores)}
    
    def get_full_attention_matrix(
        self, 
        x_num: Optional[Tensor], 
        x_cat: Optional[Tensor]
    ) -> Tensor:
        """Récupère la matrice d'attention complète pour l'analyse des interactions.
        
        Args:
            x_num: features numériques
            x_cat: features catégorielles
            
        Returns:
            ndarray: matrice d'attention de forme (seq_len, seq_len)
        """
        with torch.no_grad():
            _, last_attention = self.forward(x_num, x_cat, return_attention=True)
        avg_attention = last_attention.mean(0)  # Moyenner sur le batch
        return avg_attention.cpu().numpy()
    
    def get_attention_heatmap(
        self, 
        x_num: Optional[Tensor], 
        x_cat: Optional[Tensor], 
        include_feature_interactions: bool = False
    ) -> Union[Dict[str, float], Tensor]:
        """Interface unifiée pour récupérer les données d'attention.
        
        Args:
            x_num: features numériques
            x_cat: features catégorielles
            include_feature_interactions: si True, retourne la matrice complète
            
        Returns:
            dict ou ndarray: importance CLS ou matrice complète selon le paramètre
        """
        if include_feature_interactions:
            return self.get_full_attention_matrix(x_num, x_cat)
        else:
            return self.get_cls_importance(x_num, x_cat)
    
    
    def optimization_param_groups(self) -> List[Dict[str, Any]]:
        """Groupes de paramètres optimisés pour l'entraînement.
        
        Suit la stratégie RTDL de différenciation du weight decay selon
        le type de paramètres (embedding, normalisation, biais).
        
        Returns:
            list: groupes de paramètres avec configurations de weight decay
            
        Example:
            .. testcode::
            
                optimizer = torch.optim.AdamW(
                    model.optimization_param_groups(), 
                    lr=1e-4, 
                    weight_decay=1e-5
                )
        """
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
    
    def make_default_optimizer(self) -> torch.optim.AdamW:
        """Crée l'optimiseur par défaut avec configuration RTDL.
        
        Returns:
            AdamW: optimiseur configuré avec les meilleures pratiques
        """
        return torch.optim.AdamW(
            self.optimization_param_groups(),
            lr=1e-4,
            weight_decay=1e-5,
        )