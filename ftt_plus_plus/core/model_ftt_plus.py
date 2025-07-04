"""
Wrapper Générique pour le Modèle FTT+

Ce module contient un wrapper générique pour interfacer avec le modèle FTT+
existant dans le module ftt_plus. Aucune duplication de code, juste une interface
propre pour l'utilisation dans le pipeline FTT++.
"""

import torch
from typing import Dict, List, Optional, Any, Tuple
from torch import Tensor

# Import du modèle FTT+ existant
from ftt_plus.model import InterpretableFTTPlus


class FTTPlusModelWrapper:
    """
    Wrapper générique pour le modèle FTT+ existant.
    
    Cette classe fournit simplement une interface
    standardisée pour l'utilisation du modèle FTT+ dans le pipeline FTT++.
    """
    
    def __init__(self, model: InterpretableFTTPlus, feature_names: List[str]):
        """
        Args:
            model: Instance du modèle FTT+ pré-configuré
            feature_names: Liste des noms de features pour l'interprétabilité
        """
        self.model = model
        self.feature_names = feature_names
        self.device = next(model.parameters()).device
    
    @classmethod
    def create_model(
        cls,
        n_num_features: int,
        cat_cardinalities: List[int],
        feature_names: List[str],
        model_config: Dict[str, Any],
        device: str = 'cuda'
    ) -> 'FTTPlusModelWrapper':
        """
        Crée un nouveau modèle FTT+ avec la configuration donnée.
        
        Args:
            n_num_features: Nombre de features numériques
            cat_cardinalities: Cardinalités des features catégorielles
            feature_names: Noms des features
            model_config: Configuration du modèle
            device: Device à utiliser
            
        Returns:
            Wrapper du modèle FTT+ configuré
        """
        # Créer le modèle FTT+ avec la méthode baseline
        model = InterpretableFTTPlus.make_baseline(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            **model_config
        )
        
        model.to(device)
        
        return cls(model, feature_names)
    
    def forward(self, x_num: Optional[Tensor], x_cat: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Forward pass du modèle.
        
        Args:
            x_num: Features numériques
            x_cat: Features catégorielles
            
        Returns:
            logits: Scores de prédiction
            attention: Poids d'attention de la dernière couche
        """
        return self.model(x_num, x_cat)
    
    def get_cls_importance(
        self, 
        x_num: Optional[Tensor], 
        x_cat: Optional[Tensor]
    ) -> Dict[str, float]:
        """
        Calcule l'importance des features basée sur l'attention CLS.
        
        Args:
            x_num: Features numériques
            x_cat: Features catégorielles
            
        Returns:
            Dict mapping feature_name -> importance_score
        """
        return self.model.get_cls_importance(x_num, x_cat, self.feature_names)
    
    def get_attention_heatmap(
        self, 
        x_num: Optional[Tensor], 
        x_cat: Optional[Tensor], 
        include_feature_interactions: bool = False
    ):
        """
        Interface unifiée pour récupérer les données d'attention.
        
        Args:
            x_num: Features numériques
            x_cat: Features catégorielles
            include_feature_interactions: Si True, retourne la matrice complète
            
        Returns:
            Dict ou ndarray selon include_feature_interactions
        """
        return self.model.get_attention_heatmap(x_num, x_cat, include_feature_interactions)
    
    def parameters(self):
        """Retourne les paramètres du modèle."""
        return self.model.parameters()
    
    def named_parameters(self):
        """Retourne les paramètres nommés du modèle."""
        return self.model.named_parameters()
    
    def optimization_param_groups(self) -> List[Dict[str, Any]]:
        """Groupes de paramètres optimisés pour l'entraînement."""
        return self.model.optimization_param_groups()
    
    def train(self):
        """Met le modèle en mode entraînement."""
        self.model.train()
    
    def eval(self):
        """Met le modèle en mode évaluation."""
        self.model.eval()
    
    def to(self, device):
        """Déplace le modèle vers le device spécifié."""
        self.model.to(device)
        self.device = device
        return self
    
    def state_dict(self):
        """Retourne l'état du modèle."""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Charge l'état du modèle."""
        return self.model.load_state_dict(state_dict)
    
    def configure_num_embedding(self, embedding_type: str, X_train: Tensor, d_embedding: int, y_train: Optional[Tensor] = None):
        """
        Configure l'embedding numérique personnalisé.
        
        Args:
            embedding_type: Type d'embedding (ex: "LR", "P-LR-LR", etc.)
            X_train: Données d'entraînement pour l'embedding
            d_embedding: Dimension de l'embedding
            y_train: Labels d'entraînement si nécessaire
        """
        from num_embedding_factory import get_num_embedding
        
        num_embedding = get_num_embedding(
            embedding_type=embedding_type,
            X_train=X_train,
            d_embedding=d_embedding,
            y_train=y_train if embedding_type in ("T", "T-L", "T-LR", "T-LR-LR") else None
        )
        
        self.model.feature_tokenizer.num_tokenizer = num_embedding
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retourne des informations sur le modèle.
        
        Returns:
            Dict avec les informations du modèle
        """
        n_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            'model_type': 'FTT+',
            'n_parameters': n_params,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'device': str(self.device)
        }
    
    def __call__(self, x_num: Optional[Tensor], x_cat: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        """Permet d'appeler le wrapper comme une fonction."""
        return self.forward(x_num, x_cat)