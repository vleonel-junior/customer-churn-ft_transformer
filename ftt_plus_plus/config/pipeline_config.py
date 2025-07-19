from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class FTTPlusPlusConfig:
    """
    Configuration générique pour le pipeline FTT++.
    
    Cette classe contient uniquement des paramètres génériques
    sans aucune référence à un dataset spécifique.
    """
    
    # Étape 1: Configuration FTT+
    ftt_plus_config: Dict[str, Any]
    
    # Étape 2: Configuration du modèle Random
    M: int  # Nombre de features à sélectionner
    k: int  # Nombre d'interactions feature-feature aléatoires
    random_model_config: Dict[str, Any]
    
    # Configuration générale
    attention_seed: Optional[int] = 42
    results_dir: str = 'results'
    save_intermediate: bool = True
    
    def __post_init__(self):
        """Validation de la configuration."""
        if self.M <= 0:
            raise ValueError(f"M doit être positif, reçu: {self.M}")
        
        if self.k < 0:
            raise ValueError(f"k doit être positif ou nul, reçu: {self.k}")
        
        # Validation des configs de modèles
        required_ftt_plus_keys = ['d_token', 'n_blocks', 'd_out', 'n_heads']
        for key in required_ftt_plus_keys:
            if key not in self.ftt_plus_config:
                raise ValueError(f"Clé manquante dans ftt_plus_config: {key}")
        
        required_random_keys = ['d_token', 'n_blocks', 'd_out', 'n_heads']
        for key in required_random_keys:
            if key not in self.random_model_config:
                raise ValueError(f"Clé manquante dans random_model_config: {key}")
    
    @classmethod
    def create_default_config(
        cls,
        M: int,
        k: int,
        d_token: int = 64,
        n_blocks: int = 3,
        n_heads: int = 8,
        attention_seed: Optional[int] = 42,
        results_dir: str = 'results'
    ) -> 'FTTPlusPlusConfig':
        """
        Crée une configuration par défaut.
        
        Args:
            M: Nombre de features à sélectionner
            k: Nombre d'interactions aléatoires
            d_token: Dimension des tokens
            n_blocks: Nombre de blocs Transformer
            n_heads: Nombre de têtes d'attention
            attention_seed: Seed pour l'attention
            results_dir: Répertoire de résultats
            
        Returns:
            FTTPlusPlusConfig: Configuration par défaut
        """
        # Configuration FTT+ par défaut
        ftt_plus_config = {
            'd_token': d_token,
            'n_blocks': n_blocks,
            'n_heads': n_heads,
            'attention_dropout': 0.1,
            'ffn_d_hidden': d_token * 2,
            'ffn_dropout': 0.1,
            'residual_dropout': 0.1,
            'd_out': 1
        }
        
        # Configuration Random par défaut (similaire à FTT+)
        random_model_config = {
            'd_token': d_token,
            'n_blocks': n_blocks,
            'n_heads': n_heads,
            'attention_dropout': 0.1,
            'ffn_d_hidden': d_token * 2,
            'ffn_dropout': 0.1,
            'residual_dropout': 0.1,
            'd_out': 1
        }
        
        return cls(
            ftt_plus_config=ftt_plus_config,
            M=M,
            k=k,
            random_model_config=random_model_config,
            attention_seed=attention_seed,
            results_dir=results_dir,
            save_intermediate=True
        )
    
    def get_max_random_pairs(self, M_actual: int) -> int:
        """
        Calcule le nombre maximum de paires possibles pour M features.
        
        Args:
            M_actual: Nombre effectif de features sélectionnées
            
        Returns:
            Nombre maximum de paires possibles
        """
        return M_actual * (M_actual - 1)  # Sans auto-attention
    
    def validate_k_against_M(self, M_actual: int):
        """
        Valide que k n'est pas trop grand par rapport à M.
        
        Args:
            M_actual: Nombre effectif de features sélectionnées
        """
        max_pairs = self.get_max_random_pairs(M_actual)
        if self.k > max_pairs:
            print(f"⚠️  Attention: k={self.k} > max_pairs={max_pairs} pour M={M_actual}")
            print(f"   Le nombre de paires sera limité à {max_pairs}")
    
    def copy_with_overrides(self, **overrides) -> 'FTTPlusPlusConfig':
        """
        Crée une copie de la configuration avec des modifications.
        
        Args:
            **overrides: Paramètres à modifier
            
        Returns:
            Nouvelle instance avec les modifications
        """
        # Créer un dictionnaire avec tous les champs actuels
        current_values = {
            'ftt_plus_config': self.ftt_plus_config.copy(),
            'M': self.M,
            'k': self.k,
            'random_model_config': self.random_model_config.copy(),
            'attention_seed': self.attention_seed,
            'results_dir': self.results_dir,
            'save_intermediate': self.save_intermediate
        }
        
        # Appliquer les modifications
        current_values.update(overrides)
        
        return self.__class__(**current_values)