"""
Module de Visualisation  pour FTT+ Interprétable

Deux fonctions principales qui utilisent get_attention_heatmap() du modèle:
1. visualize_cls_importance() - Pour l'importance CLS → features
2. visualize_full_interactions() - Pour toutes les interactions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Union

def create_importance_bar_chart(
    importance_data: Union[Dict, np.ndarray],
    feature_names: List[str] = None,
    output_path: str = None,
    title: str = 'Importance des Features',
    show_annotations: bool = True
):
    """
    Fonction pour créer un graphique en barres d'importance
    """
    # Normaliser l'entrée
    if isinstance(importance_data, dict):
        feature_names = list(importance_data.keys())
        scores_array = np.array(list(importance_data.values()))
    elif isinstance(importance_data, np.ndarray):
        if feature_names is None:
            raise ValueError("feature_names requis pour np.ndarray")
        scores_array = importance_data
    else:
        raise TypeError(f"Type non supporté: {type(importance_data)}")
    
    # Créer la visualisation
    plt.figure(figsize=(16, 6))
    sns.set_style("whitegrid")
    
    # Trier par importance décroissante
    sorted_data = sorted(zip(feature_names, scores_array), key=lambda x: x[1], reverse=True)
    names, scores = zip(*sorted_data)
    
    # Graphique en barres
    bars = plt.bar(range(len(names)), scores, color='steelblue', alpha=0.8)
    
    # Mise en forme
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Score d\'importance', fontsize=12)
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Annotations pour toutes les barres
    if show_annotations:
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.annotate(f'{score:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1, format='png')
        plt.close()
        print(f"Graphique d'importance sauvegardé: {output_path}")
    else:
        plt.show()

def visualize_cls_importance(
    model,
    x_num,
    x_cat,
    feature_names: List[str],
    output_path: str,
    title: str = 'Importance CLS → Features'
):
    """
    Visualise l'importance CLS → features en utilisant get_attention_heatmap()
    """
    # Récupérer les scores CLS via get_attention_heatmap
    cls_scores = model.get_attention_heatmap(x_num, x_cat, include_feature_interactions=False)
    create_importance_bar_chart(cls_scores, feature_names, output_path, title)

def create_interactions_heatmap_from_matrix(
    attention_matrix: np.ndarray,
    feature_names_with_cls: List[str],
    output_path: str,
    title: str = 'Matrice Complète des Interactions'
):
    """
    Crée une heatmap à partir d'une matrice d'attention déjà calculée
    """
    plt.figure(figsize=(12, 10))
    sns.set_style("white")
    
    ax = sns.heatmap(
        attention_matrix,
        xticklabels=feature_names_with_cls,
        yticklabels=feature_names_with_cls,
        cmap='RdYlBu_r',
        annot=True,
        fmt='.3f',
        square=True,
        cbar_kws={
            'label': 'Score d\'attention',
            'shrink': 0.8
        }
    )
    
    # Mise en forme
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    # Highlighting CLS
    ax.axhline(y=0.5, color='red', linewidth=2, alpha=0.6)
    ax.axvline(x=0.5, color='red', linewidth=2, alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1, format='png')
    plt.close()
    
    print(f"Heatmap interactions sauvegardée: {output_path}")

def visualize_full_interactions(
    model,
    x_num,
    x_cat,
    feature_names: List[str],
    output_path: str,
    title: str = 'Matrice Complète des Interactions'
):
    """
    Visualise toutes les interactions via get_attention_heatmap()
    """
    # Récupérer la matrice complète via get_attention_heatmap
    full_matrix = model.get_attention_heatmap(x_num, x_cat, include_feature_interactions=True)
    
    # Utiliser la fonction réutilisable
    create_interactions_heatmap_from_matrix(
        full_matrix,
        ['CLS'] + feature_names,
        output_path,
        title
    )