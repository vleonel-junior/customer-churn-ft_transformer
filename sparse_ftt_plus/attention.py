"""
Mécanisme d'Attention Sélective et Flexible pour Données Tabulaires avec Sparsemax

Ce module implémente plusieurs schémas d'attention adaptés aux données tabulaires :
- 'cls'   : attention uniquement entre le token CLS et les features (FTT+ original)
- 'full'  : attention complète entre toutes les positions (hors diagonale)
- 'hybrid': attention CLS↔features et features↔features (hors diagonale)

Cette flexibilité permet d'explorer l'impact des interactions entre features sur la performance
et l'interprétabilité du modèle, au-delà du schéma FTT+ original.

Principe fondamental (Isomura et al.):
------------------------------------
"Les relations entre variables d'entrée et de sortie dans les données tabulaires
sont généralement moins complexes que dans les données non structurées. Une attention
excessive sur toutes les relations entre caractéristiques peut être non seulement
inutile mais potentiellement nuisible à la performance."

1. Contrairement aux données textuelles ou images, les relations entre caractéristiques
   dans les données tabulaires sont généralement moins complexes et moins interdépendantes.

2. Un calcul exhaustif de l'attention entre toutes les caractéristiques n'est pas
   toujours nécessaire et peut même nuire à la performance du modèle.

3. Les relations les plus importantes se situent entre le token CLS et les autres
   caractéristiques, puisque le token CLS est utilisé pour la prédiction finale.

Caractéristiques principales:
----------------------------
- Attention flexible : modes 'cls', 'full', 'hybrid' sélectionnables
- Attention focalisée sur le token CLS ou interactions complètes selon le mode
- Suppression de l'attention diagonale (self-attention)
- Réduction de la complexité computationnelle selon le mode choisi
- Utilisation de sparsemax pour une interprétabilité accrue
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Importation de sparsemax - nécessite l'installation de sparsemax
# pip install sparsemax
from sparsemax import Sparsemax
sparsemax = Sparsemax(dim=-1)

class SelectiveAttention(nn.Module):
    """
    Mécanisme d'attention sélective pour FTT+ qui:
    1. Se concentre uniquement sur les interactions avec le token CLS
    2. Évite l'attention diagonale (self-attention)
    """
    def __init__(self, d_model, n_heads, dropout=0.1, initialization='kaiming'):
        super().__init__()
        assert d_model % n_heads == 0, "d_model doit être divisible par n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        
        # Projections linéaires pour Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        # Initialisation
        if initialization == 'kaiming':
            for proj in [self.q_proj, self.k_proj, self.v_proj, self.output_proj]:
                nn.init.kaiming_normal_(proj.weight)
                nn.init.zeros_(proj.bias)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def _shape(self, tensor, seq_len, batch_size):
        """Reshape tensor pour l'attention multi-têtes"""
        return tensor.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor de forme (batch_size, seq_len, d_model)
               où seq_len = 1 + n_features (CLS token + features)
            mask: Masque d'attention optionnel
        """
        batch_size, seq_len, _ = x.shape
        
        # Projections Q, K, V
        q = self._shape(self.q_proj(x), seq_len, batch_size)  # (batch, n_heads, seq_len, d_head)
        k = self._shape(self.k_proj(x), seq_len, batch_size)  # (batch, n_heads, seq_len, d_head)
        v = self._shape(self.v_proj(x), seq_len, batch_size)  # (batch, n_heads, seq_len, d_head)
        
        # Calculer les scores d'attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # (batch, n_heads, seq_len, seq_len)
        
        # Créer un masque pour:
        # 1. Garder uniquement l'attention avec le token CLS (première position)
        # 2. Éviter l'attention diagonale
        cls_mask = torch.zeros_like(scores, dtype=torch.bool)
        cls_mask[:, :, 0, :] = True  # Permettre au CLS token d'interagir avec toutes les features
        cls_mask[:, :, :, 0] = True  # Permettre à toutes les features d'interagir avec le CLS token
        
        # Masquer l'attention diagonale
        diagonal_mask = torch.eye(seq_len, dtype=torch.bool, device=scores.device)
        diagonal_mask = diagonal_mask.unsqueeze(0).unsqueeze(0)  # Ajouter les dimensions batch et n_heads
        cls_mask = cls_mask & ~diagonal_mask
        
        # Appliquer le masque
        scores = scores.masked_fill(~cls_mask, float('-inf'))
        
        # Appliquer le masque additionnel si fourni
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Sparsemax et dropout
        attention_weights = sparsemax(scores)
        attention_weights = self.dropout_layer(attention_weights)
        
        # Calculer la sortie
        output = torch.matmul(attention_weights, v)  # (batch, n_heads, seq_len, d_head)
        output = output.transpose(1, 2).contiguous()  # (batch, seq_len, n_heads, d_head)
        output = output.view(batch_size, seq_len, self.d_model)  # (batch, seq_len, d_model)
        
        return self.output_proj(output), attention_weights

    def get_attention_weights(self, x):
        """
        Récupérer les poids d'attention pour l'interprétabilité
        """
        with torch.no_grad():
            _, attention_weights = self.forward(x)
        return attention_weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Mécanisme d'Attention Multi-Têtes Interprétable inspiré du TFT avec Sparsemax
    
    Innovation Principale:
    ---------------------
    - Partage de la matrice V entre toutes les têtes d'attention
    - Moyennage des scores d'attention pour une interprétabilité directe
    - Les poids d'attention reflètent directement l'importance réelle des features
    - Utilisation de sparsemax au lieu de softmax pour une attention plus creuse
    
    Principe Fondamental:
    --------------------
    Contrairement au multi-head attention classique où chaque tête a ses propres
    matrices Q, K, V, cette implémentation:
    1. Garde des matrices Q, K spécifiques à chaque tête (pour la diversité)
    2. Partage une seule matrice V entre toutes les têtes (pour l'interprétabilité)
    3. Moyenne les scores d'attention de toutes les têtes
    4. Applique sparsemax aux scores moyennés
    
    Résultat: Les poids d'attention moyennés représentent directement l'importance
    réelle de chaque feature, permettant une interprétation claire des heatmaps.
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1, initialization='kaiming', attention_mode='hybrid'):
        super().__init__()
        assert d_model % n_heads == 0, "d_model doit être divisible par n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        self.attention_mode = attention_mode  # 'cls', 'full', 'hybrid'
        
        # Projections Q et K spécifiques à chaque tête (pour la diversité)
        self.q_projections = nn.ModuleList([
            nn.Linear(d_model, self.d_head) for _ in range(n_heads)
        ])
        self.k_projections = nn.ModuleList([
            nn.Linear(d_model, self.d_head) for _ in range(n_heads)
        ])
        
        # Projection V PARTAGÉE (clé de l'interprétabilité)
        self.v_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        # Initialisation
        if initialization == 'kaiming':
            for q_proj in self.q_projections:
                nn.init.kaiming_normal_(q_proj.weight)
                nn.init.zeros_(q_proj.bias)
            for k_proj in self.k_projections:
                nn.init.kaiming_normal_(k_proj.weight)
                nn.init.zeros_(k_proj.bias)
            nn.init.kaiming_normal_(self.v_proj.weight)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.kaiming_normal_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor de forme (batch_size, seq_len, d_model)
               où seq_len = 1 + n_features (CLS token + features)
            mask: Masque d'attention optionnel
        
        Returns:
            output: Tensor transformé de forme (batch_size, seq_len, d_model)
            avg_attention: Scores d'attention moyennés pour interprétabilité
        """
        batch_size, seq_len, _ = x.shape
        
        # Calculer la matrice V partagée
        v = self.v_proj(x)  # (batch_size, seq_len, d_model)
        
        # Calculer les scores d'attention pour chaque tête
        attention_scores_per_head = []
        
        for h in range(self.n_heads):
            # Projections Q et K spécifiques à cette tête
            q_h = self.q_projections[h](x)  # (batch_size, seq_len, d_head)
            k_h = self.k_projections[h](x)  # (batch_size, seq_len, d_head)
            
            # Calculer les scores d'attention pour cette tête
            scores_h = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(self.d_head)
            # (batch_size, seq_len, seq_len)
            
            # Créer le masque sélectif selon le mode choisi
            if self.attention_mode == 'cls':
                cls_mask = torch.zeros_like(scores_h, dtype=torch.bool)
                cls_mask[:, 0, :] = True  # CLS peut regarder toutes les features
                cls_mask[:, :, 0] = True  # Toutes les features peuvent regarder CLS
                
                # Masquer l'attention diagonale
                diagonal_mask = torch.eye(seq_len, dtype=torch.bool, device=scores_h.device)
                diagonal_mask = diagonal_mask.unsqueeze(0)  # Ajouter dimension batch
                cls_mask = cls_mask & ~diagonal_mask
                
                # Appliquer le masque
                scores_h = scores_h.masked_fill(~cls_mask, float('-inf'))
            elif self.attention_mode == 'full':
                # Autoriser toute l'attention sauf diagonale
                full_mask = ~torch.eye(seq_len, dtype=torch.bool, device=scores_h.device).unsqueeze(0)
                scores_h = scores_h.masked_fill(~full_mask, float('-inf'))
            elif self.attention_mode == 'hybrid':
                # Autoriser CLS<->features et features<->features (sauf diagonale)
                hybrid_mask = torch.ones_like(scores_h, dtype=torch.bool)
                diagonal_mask = torch.eye(seq_len, dtype=torch.bool, device=scores_h.device).unsqueeze(0)
                hybrid_mask = hybrid_mask & ~diagonal_mask
                scores_h = scores_h.masked_fill(~hybrid_mask, float('-inf'))
            else:
                raise ValueError(f"attention_mode inconnu: {self.attention_mode}")
            
            # Appliquer le masque additionnel si fourni
            if mask is not None:
                scores_h = scores_h.masked_fill(~mask, float('-inf'))
            
            # Appliquer sparsemax pour cette tête (pour conserver la diversité des têtes et avoir une attention creuse)
            attention_h = sparsemax(scores_h)
            attention_scores_per_head.append(attention_h)
        
        # INNOVATION: Moyenner les scores d'attention de toutes les têtes
        # C'est ici que réside l'interprétabilité directe
        avg_attention = torch.stack(attention_scores_per_head, dim=0).mean(dim=0)
        # (batch_size, seq_len, seq_len)
        
        # Appliquer le dropout sur l'attention moyennée
        # avg_attention = self.dropout_layer(avg_attention)
        
        # Calculer la sortie en utilisant l'attention moyennée et la matrice V partagée
        output = torch.matmul(avg_attention, v)  # (batch_size, seq_len, d_model)
        
        # Projection finale
        output = self.output_proj(output)
        
        return output, avg_attention
        # (batch_size, seq_len, seq_len)