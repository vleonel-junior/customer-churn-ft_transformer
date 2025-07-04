from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class FeatureMapping:
    """
    Cette classe permet de définir un mapping flexible entre les features
    numériques et catégorielles, tout en assurant la cohérence des données.
    """
    
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
    def create_mapping(
        cls, 
        num_feature_names: List[str], 
        cat_feature_names: List[str]
    ) -> 'FeatureMapping':
        """
        Crée un mapping générique à partir des listes de noms.
        
        Args:
            num_feature_names: Liste des noms de features numériques
            cat_feature_names: Liste des noms de features catégorielles
            
        Returns:
            FeatureMapping: Instance configurée
        """
        all_features = num_feature_names + cat_feature_names
        
        return cls(
            num_feature_names=num_feature_names,
            cat_feature_names=cat_feature_names,
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
        """
        Valide que les données correspondent au mapping.
        
        Args:
            X_num_shape: Nombre de features numériques dans les données
            X_cat_shape: Nombre de features catégorielles dans les données
        """
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
    
    def get_feature_type(self, feature_name: str) -> str:
        """
        Détermine le type d'une feature.
        
        Args:
            feature_name: Nom de la feature
            
        Returns:
            'NUM' ou 'CAT' selon le type
        """
        if feature_name in self.num_feature_names:
            return 'NUM'
        elif feature_name in self.cat_feature_names:
            return 'CAT'
        else:
            raise ValueError(f"Feature '{feature_name}' introuvable dans le mapping")
    
    @property
    def n_num_features(self) -> int:
        """Nombre de features numériques."""
        return len(self.num_feature_names)
    
    @property
    def n_cat_features(self) -> int:
        """Nombre de features catégorielles."""
        return len(self.cat_feature_names)
    
    @property
    def n_total_features(self) -> int:
        """Nombre total de features."""
        return len(self.all_feature_names)