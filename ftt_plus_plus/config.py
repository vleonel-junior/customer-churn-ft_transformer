"""
Configuration et Mapping des Features pour FTT++

Ce module contient :
1. FTTPlusPlusConfig - Configuration complète du pipeline
2. FeatureMapping - Gestion explicite du mapping des features
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class FTTPlusPlusConfig:
    """Configuration pour le pipeline FTT++."""
    
    # Étape 1: Configuration FTT+
    ftt_plus_config: Dict[str, Any]
    
    # Étape 2: Configuration du modèle Random
    M: int  # Nombre de features à sélectionner
    k: int  # Nombre d'interactions feature-feature aléatoires
    random_model_config: Dict[str, Any]
    
    # Configuration générale
    attention_seed: Optional[int] = 42
    results_dir: str = 'results/results_telecom'
    save_intermediate: bool = True


@dataclass
class FeatureMapping:
    """Structure pour gérer explicitement le mapping des features."""
    
    num_feature_names: List[str]  # Noms des features numériques
    cat_feature_names: List[str]  # Noms des features catégorielles
    all_feature_names: List[str]  # Tous les noms dans l'ordre
    
    def __post_init__(self):
        """Validation de la cohérence du mapping."""
        expected_all = self.num_feature_names + self.cat_feature_names
        if self.all_feature_names != expected_all:
            raise ValueError(
                f"Incohérence dans le mapping des features.\n"
                f"Attendu: {expected_all}\n"
                f"Reçu: {self.all_feature_names}"
            )
    
    @classmethod
    def from_telecom_dataset(cls) -> 'FeatureMapping':
        """Crée le mapping pour le dataset Telecom."""
        num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        cat_features = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod'
        ]
        all_features = num_features + cat_features
        
        return cls(
            num_feature_names=num_features,
            cat_feature_names=cat_features,
            all_feature_names=all_features
        )
    
    def get_selected_feature_indices(self, selected_features: List[str]) -> Tuple[List[int], List[int]]:
        """
        Détermine les indices des features sélectionnées de façon robuste.
        
        Args:
            selected_features: Liste des noms de features sélectionnées
            
        Returns:
            (indices_num, indices_cat): Indices dans les arrays numériques et catégoriels
        """
        # Vérifier que toutes les features sélectionnées existent
        missing_features = [f for f in selected_features if f not in self.all_feature_names]
        if missing_features:
            raise ValueError(f"Features sélectionnées introuvables: {missing_features}")
        
        # Séparer par type de feature
        selected_num = [f for f in selected_features if f in self.num_feature_names]
        selected_cat = [f for f in selected_features if f in self.cat_feature_names]
        
        # Obtenir les indices dans chaque array
        indices_num = [self.num_feature_names.index(f) for f in selected_num]
        indices_cat = [self.cat_feature_names.index(f) for f in selected_cat]
        
        # Debug info
        print(f"📋 Mapping des features sélectionnées:")
        print(f"   - Numériques sélectionnées: {selected_num} → indices {indices_num}")
        print(f"   - Catégorielles sélectionnées: {selected_cat} → indices {indices_cat}")
        
        return indices_num, indices_cat
    
    def validate_data_consistency(self, X_num_shape: int, X_cat_shape: int):
        """Valide que les données correspondent au mapping."""
        if X_num_shape != len(self.num_feature_names):
            raise ValueError(
                f"Nombre de features numériques incohérent: "
                f"données={X_num_shape}, mapping={len(self.num_feature_names)}"
            )
        
        if X_cat_shape != len(self.cat_feature_names):
            raise ValueError(
                f"Nombre de features catégorielles incohérent: "
                f"données={X_cat_shape}, mapping={len(self.cat_feature_names)}"
            )
        
        print(f"✅ Validation mapping: {X_num_shape} num + {X_cat_shape} cat = {len(self.all_feature_names)} total")