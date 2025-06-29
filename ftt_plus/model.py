"""
Modèle FTT+ avec Attention Multi-Têtes Interprétable

Cette implémentation combine:
1. L'efficacité de l'attention sélective FTT+
2. L'interprétabilité de l'attention multi-têtes partagée (inspirée du TFT)
"""

import torch
import torch.nn as nn
from .attention import InterpretableMultiHeadAttention

from rtdl_lib.modules import FeatureTokenizer
from rtdl_lib.nn._backbones import FeedForward

class TransformerLayer(nn.Module):
    """
    Couche Transformer avec Attention Multi-Têtes Interprétable
    """
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embedding_size)
        self.attention = InterpretableMultiHeadAttention(
            d_model=config.embedding_size,
            n_heads=config.n_heads,
            dropout=config.attention_dropout,
            initialization='kaiming'
        )
        self.norm2 = nn.LayerNorm(config.embedding_size)
        self.ffn = FeedForward(
            d_token=config.embedding_size,
            d_hidden=config.ffn_hidden,
            dropout=config.ffn_dropout,
            activation='reglu'
        )
        self.residual_dropout = nn.Dropout(config.residual_dropout)

    def forward(self, x):
        # Attention avec connexion résiduelle
        residual = x
        x = self.norm1(x)
        x, attention_weights = self.attention(x)
        x = self.residual_dropout(x) + residual

        # Feed-forward avec connexion résiduelle
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.residual_dropout(x) + residual

        return x, attention_weights

class InterpretableFTTPlus(nn.Module):
    """
    Modèle FTT+ avec Attention Multi-Têtes Interprétable
    """
    def __init__(self, config, n_num_features, cat_cardinalities):
        super().__init__()
        self.config = config
        self.n_num_features = n_num_features
        self.cat_cardinalities = cat_cardinalities
        
        # Utilisation du Feature Tokenizer RTDL
        self.feature_tokenizer = FeatureTokenizer(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_token=config.embedding_size,
            initialization='uniform',
            bias=False,
            num_special_tokens=1  # Pour le token CLS
        )
        
        # Couches Transformer Interprétables
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.n_blocks)
        ])
        
        # Couche de sortie
        self.norm = nn.LayerNorm(config.embedding_size)
        self.head = nn.Sequential(
            nn.Linear(config.embedding_size, config.embedding_size),
            nn.ReLU(),
            nn.LayerNorm(config.embedding_size),
            nn.Linear(config.embedding_size, 1)
        )
        
    def forward(self, x_num, x_cat):
        # Utiliser le Feature Tokenizer modulaire pour créer la séquence complète
        x = self.feature_tokenizer(x_num=x_num, x_cat=x_cat)
        # x shape: (batch_size, seq_len, d_embedding) - inclut déjà le token CLS
        
        # Passer à travers les couches Transformer Interprétables
        for i, layer in enumerate(self.layers):
            x, attn = layer(x)
            # Stocker seulement l'attention de la dernière couche
            if i == len(self.layers) - 1:
                last_attention = attn
        
        # Normalisation finale
        x = self.norm(x)
        
        # Classification à partir du token CLS
        x = x[:, 0]  # Prendre uniquement le token CLS
        x = self.head(x)
        
        return x, last_attention
    
    def get_cls_importance(self, x_num, x_cat, feature_names=None):
        """
        Calcule l'importance des features basée sur l'attention CLS uniquement
        """
        with torch.no_grad():
            _, last_attention = self.forward(x_num, x_cat)
        
        avg_attention = last_attention.mean(0)  # (seq, seq)
        importance_scores = avg_attention[0, 1:].cpu().numpy()
        
        if feature_names is not None:
            return dict(zip(feature_names, importance_scores))
        else:
            return {f'feature_{i}': score for i, score in enumerate(importance_scores)}
    
    def get_full_attention_matrix(self, x_num, x_cat):
        """
        Récupère la matrice d'attention complète pour l'analyse des relations intra-variables
        """
        with torch.no_grad():
            _, last_attention = self.forward(x_num, x_cat)
        avg_attention = last_attention.mean(0)  # (seq_len, seq_len)
        return avg_attention.cpu().numpy()
    
    def get_attention_heatmap(self, x_num, x_cat, include_feature_interactions=False):
        """
        Récupère les données d'attention pour visualisation
        """
        if include_feature_interactions:
            return self.get_full_attention_matrix(x_num, x_cat)
        else:
            return self.get_cls_importance(x_num, x_cat)
    
    def explain_prediction(self, x_num, x_cat, feature_names, threshold=0.1):
        """
        Fournit une explication complète de la prédiction
        """
        with torch.no_grad():
            logits, _ = self.forward(x_num.unsqueeze(0), x_cat.unsqueeze(0))
            prediction = torch.sigmoid(logits).item()
        importance = self.get_cls_importance(
            x_num.unsqueeze(0), x_cat.unsqueeze(0), feature_names
        )
        important_features = {
            name: score for name, score in importance.items() 
            if score > threshold
        }
        important_features = dict(
            sorted(important_features.items(), key=lambda x: x[1], reverse=True)
        )
        return {
            'prediction': prediction,
            'prediction_label': 'Churn' if prediction > 0.5 else 'No Churn',
            'confidence': max(prediction, 1 - prediction),
            'important_features': important_features,
            'all_feature_importance': importance
        }