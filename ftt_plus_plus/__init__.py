"""
FTT++ (FT-Transformer Plus Plus) - Approche en Deux Étapes pour l'Interprétabilité Avancée

Ce module implémente l'approche FTT++ proposée par Isomura et al. qui combine :
1. L'entraînement d'un modèle FTT+ pour identifier les features importantes
2. L'entraînement d'un modèle Random sparse sur les features sélectionnées

Cette approche en deux temps permet de focaliser l'attention sur un sous-ensemble
pertinent de caractéristiques, améliorant à la fois la performance et l'interprétabilité.

Architecture Modulaire:
    - core: Modèles et attention sparse génériques
    - config: Configuration et mapping des features génériques
    - training: Gestionnaires d'entraînement pour chaque étape
    - pipeline: Pipeline complet FTT++ (orchestration des deux étapes)
    - visualisation: Visualisations FTT++
"""

# Configuration
from .config.pipeline_config import FTTPlusPlusConfig
from .config.feature_mapping import FeatureMapping

# Core models et attention
from .core.sparse_attention import SparseRandomAttention
from .core.model_ftt_plus import FTTPlusModelWrapper
from .core.model_ftt_random import FTTRandomModel

# Training
from .training.stage1_trainer import Stage1Trainer
from .training.stage2_trainer import Stage2Trainer

# Pipeline
from .pipeline.pipeline import FTTPlusPlusPipeline

# Visualisation
from .visualisation.visualisation import (
    create_ftt_plus_plus_importance_chart,
    create_sparse_attention_heatmap,
    visualize_sparse_attention_heatmap
)


__all__ = [
    # Configuration
    'FTTPlusPlusConfig',
    'FeatureMapping',
    
    # Core models et attention
    'SparseRandomAttention',
    'FTTPlusModelWrapper',
    'FTTRandomModel',
    
    # Training
    'Stage1Trainer',
    'Stage2Trainer',
    
    # Pipeline
    'FTTPlusPlusPipeline',
    
    # Visualisation
    'create_ftt_plus_plus_importance_chart',
    'create_sparse_attention_heatmap',
    'visualize_sparse_attention_heatmap'
]

__version__ = "1.0.0"
__author__ = "Léonel VODOUNOU"
__description__ = "FT-Transformer Plus Plus - Enhanced Interpretability through Two-Stage Training"