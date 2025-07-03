"""
Module de Visualisation pour FTT++ avec Attention Sparse

Ce module adapte les fonctionnalités de ftt_plus/visualisation.py
pour les modèles FTT++ avec attention sparse.

Fonctions principales:
1. visualize_sparse_attention_heatmap() - Heatmap complète des interactions
2. create_ftt_plus_plus_importance_chart() - Graphique d'importance avec features sélectionnées
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Union, Optional

def create_ftt_plus_plus_importance_chart(
    importance_data: Union[Dict, np.ndarray],
    feature_names: List[str] = None,
    output_path: str = None,
    title: str = 'Importance des Features - FTT++',
    show_annotations: bool = True,
    highlight_selected: Optional[List[str]] = None
):
    """
    Graphique d'importance adapté pour FTT++ avec mise en évidence des features sélectionnées
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
    
    # Couleurs : mettre en évidence les features sélectionnées
    colors = []
    for name in names:
        if highlight_selected and name in highlight_selected:
            colors.append('orange')  # Features sélectionnées
        else:
            colors.append('steelblue')  # Features non sélectionnées
    
    # Graphique en barres
    bars = plt.bar(range(len(names)), scores, color=colors, alpha=0.8)
    
    # Mise en forme
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Score d\'importance', fontsize=12)
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Légende si highlight_selected est fourni
    if highlight_selected:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='orange', label=f'Features sélectionnées (M={len(highlight_selected)})'),
            Patch(facecolor='steelblue', label='Features non sélectionnées')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
    
    # Annotations
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
        print(f"Graphique d'importance FTT++ sauvegardé: {output_path}")
    else:
        plt.show()

def create_sparse_attention_heatmap(
    attention_matrix: np.ndarray,
    feature_names_with_cls: List[str],
    output_path: str,
    title: str = 'Heatmap d\'Attention Sparse - FTT++'
):
    """
    Crée une heatmap d'attention sparse à partir d'une matrice déjà calculée
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
            'label': 'Score d\'attention sparse',
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
    
    print(f"Heatmap d'attention sparse sauvegardée: {output_path}")

def visualize_sparse_attention_heatmap(
    model,
    x_num,
    x_cat,
    feature_names: List[str],
    output_path: str,
    title: str = 'Heatmap d\'Attention Sparse - FTT++'
):
    """
    Visualise la matrice d'attention sparse complète pour FTT++
    """
    # Récupérer la matrice complète via get_attention_heatmap
    full_matrix = model.get_attention_heatmap(x_num, x_cat, include_feature_interactions=True)
    
    # Utiliser la fonction réutilisable
    create_sparse_attention_heatmap(
        full_matrix,
        ['CLS'] + feature_names,
        output_path,
        title
    )