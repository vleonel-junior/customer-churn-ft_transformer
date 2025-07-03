"""
Mécanisme d'Attention Sparse pour FTT++ - Modèle Random

Ce module implémente l'attention sparse utilisée dans la deuxième étape de FTT++.
Le mécanisme combine :
1. Attention entre le token [CLS] et les M features sélectionnées
2. Attention pour k paires de features choisies aléatoirement
3. Exclusion de l'auto-attention (comme dans FTT+)

Principe:
---------
- Focus sur les interactions essentielles (CLS ↔ features importantes)
- Exploration limitée des interactions feature-feature (k paires aléatoires)
- Réduction de la complexité computationnelle
- Maintien de l'interprétabilité

Innovation:
-----------
Cette approche combine la focalisation de FTT+ avec une exploration
contrôlée des interactions internes, permettant de capturer des relations
importantes tout en gardant la sparsité et l'interprétabilité.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import List, Tuple, Optional, Set, Dict, Any


class SparseRandomAttention(nn.Module):
    """
    Mécanisme d'attention sparse avec interactions aléatoires contrôlées.
    
    Cette classe implémente l'attention utilisée dans le modèle Random de FTT++,
    qui se concentre sur :
    1. Les interactions CLS ↔ features sélectionnées
    2. k paires d'interactions feature-feature choisies aléatoirement
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        M: int,  # Nombre de features sélectionnées
        k: int,  # Nombre de paires d'interactions aléatoires
        dropout: float = 0.1,
        initialization: str = 'kaiming',
        seed: Optional[int] = None
    ):
        """
        Args:
            d_model: Dimension du modèle
            n_heads: Nombre de têtes d'attention
            M: Nombre de features sélectionnées (depuis l'étape 1)
            k: Nombre de paires d'interactions feature-feature aléatoires
            dropout: Taux de dropout
            initialization: Type d'initialisation
            seed: Seed pour la reproductibilité des paires aléatoires
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model doit être divisible par n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.M = M  # Nombre de features sélectionnées
        self.k = k  # Nombre de paires aléatoires
        self.dropout = dropout
        self.seed = seed
        
        # Projections Q, K, V par tête (comme dans l'attention TFT)
        self.q_projections = nn.ModuleList([
            nn.Linear(d_model, self.d_head) for _ in range(n_heads)
        ])
        self.k_projections = nn.ModuleList([
            nn.Linear(d_model, self.d_head) for _ in range(n_heads)
        ])
        
        # Projection V partagée (pour l'interprétabilité)
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
        
        # Cache pour les paires aléatoires (pour la reproductibilité)
        self._random_pairs_cache = None
    
    def _generate_random_pairs(self, M: int, k: int) -> List[Tuple[int, int]]:
        """
        Génère k paires d'indices aléatoires parmi les M features sélectionnées.
        
        Args:
            M: Nombre total de features sélectionnées
            k: Nombre de paires à générer
            
        Returns:
            Liste de k paires (i, j) avec i ≠ j
        """
        # Fixer le seed si fourni pour la reproductibilité
        if self.seed is not None:
            random.seed(self.seed)
        
        # Générer toutes les paires possibles (sans auto-attention)
        all_pairs = [(i, j) for i in range(1, M + 1) for j in range(1, M + 1) if i != j]
        
        # Sélectionner k paires aléatoirement
        if k >= len(all_pairs):
            return all_pairs  # Retourner toutes les paires si k est trop grand
        
        selected_pairs = random.sample(all_pairs, k)
        return selected_pairs
    
    def _create_sparse_attention_mask(self, seq_len: int) -> torch.Tensor:
        """
        Crée le masque d'attention sparse pour FTT++.
        
        Args:
            seq_len: Longueur de la séquence (1 + M, où 1 = CLS token)
            
        Returns:
            Masque booléen de forme (seq_len, seq_len)
        """
        # Initialiser le masque à False (tout masqué)
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        
        # 1. Permettre l'attention CLS ↔ toutes les features sélectionnées
        mask[0, 1:] = True  # CLS vers features
        mask[1:, 0] = True  # Features vers CLS
        
        # 2. Générer les paires aléatoires si pas déjà en cache
        if self._random_pairs_cache is None or len(self._random_pairs_cache) != self.k:
            self._random_pairs_cache = self._generate_random_pairs(self.M, self.k)
        
        # 3. Permettre l'attention pour les k paires aléatoires
        for i, j in self._random_pairs_cache:
            mask[i, j] = True
            mask[j, i] = True  # Attention bidirectionnelle
        
        # 4. L'auto-attention reste interdite (diagonale à False)
        # C'est déjà le cas car on initialise à False
        
        return mask
    
    def forward(self, x: torch.Tensor, feature_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass de l'attention sparse.
        
        Args:
            x: Tensor d'entrée de forme (batch_size, seq_len, d_model)
               où seq_len = 1 + M (CLS + M features sélectionnées)
            feature_mask: Masque additionnel optionnel
            
        Returns:
            output: Tensor transformé
            avg_attention: Poids d'attention moyennés pour interprétabilité
        """
        batch_size, seq_len, _ = x.shape
        
        # Vérifier que seq_len correspond à 1 + M
        expected_seq_len = 1 + self.M
        if seq_len != expected_seq_len:
            raise ValueError(
                f"Séquence de longueur {seq_len} mais {expected_seq_len} attendue "
                f"(1 CLS + {self.M} features sélectionnées)"
            )
        
        # Calculer la matrice V partagée
        v = self.v_proj(x)  # (batch_size, seq_len, d_model)
        
        # Créer le masque d'attention sparse
        sparse_mask = self._create_sparse_attention_mask(seq_len)
        sparse_mask = sparse_mask.to(x.device)
        
        # Calculer les scores d'attention pour chaque tête
        attention_scores_per_head = []
        
        for h in range(self.n_heads):
            # Projections Q et K spécifiques à cette tête
            q_h = self.q_projections[h](x)  # (batch_size, seq_len, d_head)
            k_h = self.k_projections[h](x)  # (batch_size, seq_len, d_head)
            
            # Calculer les scores d'attention pour cette tête
            scores_h = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(self.d_head)
            # (batch_size, seq_len, seq_len)
            
            # Appliquer le masque sparse
            scores_h = scores_h.masked_fill(~sparse_mask.unsqueeze(0), float('-inf'))
            
            # Appliquer le masque additionnel si fourni
            if feature_mask is not None:
                scores_h = scores_h.masked_fill(~feature_mask, float('-inf'))
            
            # Softmax pour cette tête
            attention_h = F.softmax(scores_h, dim=-1)
            attention_scores_per_head.append(attention_h)
        
        # Moyenner les scores d'attention de toutes les têtes (interprétabilité TFT)
        avg_attention = torch.stack(attention_scores_per_head, dim=0).mean(dim=0)
        # (batch_size, seq_len, seq_len)
        
        # Appliquer le dropout sur l'attention moyennée
        avg_attention = self.dropout_layer(avg_attention)
        
        # Calculer la sortie en utilisant l'attention moyennée et la matrice V partagée
        output = torch.matmul(avg_attention, v)  # (batch_size, seq_len, d_model)
        
        # Projection finale
        output = self.output_proj(output)
        
        return output, avg_attention
    
    def get_attention_statistics(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur le pattern d'attention sparse.
        
        Returns:
            Dictionnaire avec des informations sur la sparsité
        """
        seq_len = 1 + self.M
        total_possible_connections = seq_len * seq_len
        
        # Connexions actives
        active_connections = (
            2 * self.M +  # CLS ↔ features (bidirectionnel)
            2 * self.k    # k paires feature-feature (bidirectionnel)
        )
        
        sparsity_ratio = 1 - (active_connections / total_possible_connections)
        
        return {
            'total_features': self.M,
            'random_pairs': self.k,
            'total_connections': total_possible_connections,
            'active_connections': active_connections,
            'sparsity_ratio': sparsity_ratio,
            'cls_connections': 2 * self.M,
            'feature_feature_connections': 2 * self.k,
            'random_pairs_list': self._random_pairs_cache
        }
    
    def update_random_pairs(self, new_seed: Optional[int] = None):
        """
        Met à jour les paires aléatoires avec un nouveau seed.
        
        Args:
            new_seed: Nouveau seed (si None, utilise le seed original + 1)
        """
        if new_seed is not None:
            self.seed = new_seed
        elif self.seed is not None:
            self.seed += 1
        
        # Réinitialiser le cache pour forcer la régénération
        self._random_pairs_cache = None
    
    def visualize_attention_pattern(self) -> torch.Tensor:
        """
        Génère une matrice de visualisation du pattern d'attention sparse.
        
        Returns:
            Matrice binaire montrant les connexions autorisées
        """
        seq_len = 1 + self.M
        return self._create_sparse_attention_mask(seq_len).float()