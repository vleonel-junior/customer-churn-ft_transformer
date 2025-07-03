"""
FTT++ (FT-Transformer Plus Plus) - Approche en Deux Étapes pour l'Interprétabilité Avancée

Ce module implémente l'approche FTT++ proposée par Isomura et al. qui combine :
1. L'entraînement d'un modèle FTT+ pour identifier les features importantes
2. L'entraînement d'un modèle Random sparse sur les features sélectionnées

Cette approche en deux temps permet de focaliser l'attention sur un sous-ensemble 
pertinent de caractéristiques, améliorant à la fois la performance et l'interprétabilité.

Architecture Modulaire:
    - config: Configuration et mapping des features
    - sparse_attention: Mécanisme d'attention sparse avec interactions aléatoires
    - random_model: Modèle Random avec attention focalisée
    - training_stages: Gestionnaires d'entraînement pour chaque étape
    - pipeline: Pipeline complet FTT++ (orchestration des deux étapes)
"""

# Imports principaux pour l'interface publique
from .config import FTTPlusPlusConfig, FeatureMapping
from .sparse_attention import SparseRandomAttention
from .random_model import InterpretableFTTRandom
from .training_stages import Stage1Trainer, Stage2Trainer
from .pipeline import FTTPlusPlusPipeline

__all__ = [
    # Configuration et mapping
    'FTTPlusPlusConfig',
    'FeatureMapping',
    
    # Modèles et attention
    'SparseRandomAttention', 
    'InterpretableFTTRandom',
    
    # Entraînement
    'Stage1Trainer',
    'Stage2Trainer',
    
    # Pipeline principal
    'FTTPlusPlusPipeline'
]

__version__ = "1.0.0"
__author__ = "FTT++ Implementation Team"
__description__ = "FT-Transformer Plus Plus - Enhanced Interpretability through Two-Stage Training"