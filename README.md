# FTT+ : Feature Tokenizer Transformers interprétables pour données tabulaires

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Compatible-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

*Architecture Transformer optimisée pour l'apprentissage interprétable sur données tabulaires*

</div>

---

## Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Architecture](#architecture)
- [Variante Sparse](#variante-sparse)
- [Implémentation](#implémentation)
- [Configuration](#configuration)
- [Installation](#installation)
- [Justification scientifique](#justification-scientifique)
- [Références](#références)

---

## Vue d'ensemble

**FTT+** (FT-Transformer Plus) est une architecture Transformer spécialement conçue pour les données tabulaires, offrant une interprétabilité native sans sacrifier les performances. Le modèle propose trois modes d'attention distincts pour s'adapter aux différents besoins d'analyse.

### Modes d'attention supportés

| Mode | Description | Interactions autorisées |
|------|-------------|-------------------------|
| `cls` | Mode original | Token CLS ↔ Features uniquement |
| `hybrid` | Mode recommandé | CLS ↔ Features + Features ↔ Features |
| `full` | Mode complet | Toutes les interactions (hors diagonale) |

**Mode par défaut** : `hybrid` - optimise le rapport performance/interprétabilité.

---

## Architecture

### Pipeline général

<div align="center">
<img src="images/FT_Transformer architecture.png" alt="Architecture FTT+" width="85%" style="max-width: 750px;">
<br><br>
<em>Architecture complète du FT-Transformer pour données tabulaires</em>
</div>

### Composants principaux

#### 1. Feature Tokenizer

<div align="center">
<img src="images/Illustration%20d'un%20Feature%20Tokenizer.png" alt="Feature Tokenizer" width="85%" style="max-width: 750px;">
<br><br>
<em>Processus de tokenisation des variables en représentations vectorielles</em>
</div>

Le `FeatureTokenizer` convertit chaque variable (numérique ou catégorielle) en un vecteur dense de dimension fixe, créant une représentation homogène de l'ensemble des features.

#### 2. Blocs Transformer

<div align="center">
<img src="images/One Transformer layer.png" alt="Bloc Transformer" width="50%" style="max-width: 350px;">
<br><br>
<em>Structure d'un bloc Transformer FTT+</em>
</div>

Chaque bloc comprend :

**Interpretable Multi-Head Attention**

<div align="center">
<img src="images/Scaled Dot-Product Attention.png" alt="Mécanisme d'attention" width="50%" style="max-width: 400px;">
<br><br>
<em>Mécanisme d'attention avec masquage sélectif</em>
</div>

Caractéristiques techniques :
- Matrices Query/Key spécifiques par tête d'attention
- Matrice Value partagée entre toutes les têtes
- Moyenne des scores d'attention pour extraction directe de l'importance
- Schéma d'interaction paramétrable selon le mode choisi

<div align="center">
<img src="images/Interpretable Multi-Head Attention.png" alt="Multi-Head Attention interprétable" width="85%" style="max-width: 750px;">
<br><br>
<em>Architecture Multi-Head Attention avec extraction d'importance</em>
</div>

**Feed-Forward Network**
- Couches linéaires avec activation non-linéaire
- Normalisation par couches (LayerNorm)
- Connexions résiduelles

#### 3. Tête de classification

La prédiction finale est obtenue par projection linéaire de la représentation du token CLS.

### Extraction de l'interprétabilité

L'importance de chaque feature est calculée directement à partir des poids d'attention moyennés, permettant :
- Identification des variables les plus influentes
- Analyse des interactions entre features (modes `hybrid` et `full`)
- Extraction immédiate des scores d'importance sans post-traitement

---

## Variante Sparse

**Sparse FTT+** remplace le mécanisme d'attention softmax par sparsemax, produisant des distributions d'attention parses.

### Avantages techniques

| Aspect | Bénéfice |
|--------|----------|
| **Complexité** | Réduction du nombre d'interactions actives |
| **Interprétabilité** | Sélection explicite des features pertinents |
| **Performance** | Concentration sur les relations significatives |

### Mécanisme sparsemax

La fonction sparsemax génère des distributions où :
- Seul un sous-ensemble restreint d'éléments a des poids non-nuls
- Les interactions non-pertinentes sont explicitement éliminées (poids = 0)
- La parcimonie améliore la lisibilité des résultats d'interprétation

---

## Implémentation

### Structure du projet

```
ftt_plus/
├── attention.py         # Modules d'attention interprétable
└── model.py            # Architecture FTT+ complète

sparse_ftt_plus/
├── attention.py         # Attention sparse (sparsemax)
└── model.py            # Architecture Sparse FTT+
```

### Architecture modulaire

- **Séparation des responsabilités** : mécanismes d'attention et architecture principale
- **Extensibilité** : ajout facile de nouveaux modes d'attention
- **Réutilisabilité** : composants indépendants et paramétrables

---

## Configuration

### Paramètres FTT+

```python
ftt_plus_config = {
    # Architecture
    'd_model': 256,           # Dimension des embeddings
    'n_heads': 8,             # Nombre de têtes d'attention
    'n_layers': 6,            # Nombre de blocs Transformer
    'dropout': 0.1,           # Taux de dropout
    
    # Mode d'attention
    'attention_mode': 'hybrid',  # 'cls' | 'hybrid' | 'full'
    
    # Feature Tokenizer
    'embedding_dim': 256,     # Dimension des embeddings de features
    'categorical_embedding_dim': 32,  # Dimension pour variables catégorielles
}
```

### Recommandations par cas d'usage

- **Interprétabilité maximale** : `attention_mode='cls'`
- **Usage général** : `attention_mode='hybrid'` (défaut)
- **Analyse d'interactions complexes** : `attention_mode='full'`

---

## Installation

### Dépendances requises

```bash
pip install torch>=1.9.0 transformers numpy pandas scikit-learn
```

### Dépendances optionnelles

```bash
# Pour Sparse FTT+
pip install sparsemax
```

---

## Justification scientifique

### Problématique

Les modèles de deep learning pour données tabulaires souffrent traditionnellement d'un manque d'interprétabilité, limitant leur adoption dans des domaines critiques (finance, santé, juridique).

### Contributions de FTT+

**Innovation architecturale**
- Mécanisme d'attention spécialisé pour données structurées
- Extraction native de l'importance des features
- Flexibilité des schémas d'interaction

**Avantages par rapport aux approches existantes**
- **vs. XGBoost/Random Forest** : Capacité à modéliser des interactions complexes
- **vs. MLP standard** : Interprétabilité intégrée sans perte de performance
- **vs. Méthodes post-hoc** : Explications directes intégrées à l'architecture

### Cas d'usage cibles

- **Finance** : Scoring de crédit, détection de fraude
- **Santé** : Diagnostic assisté, analyse de biomarqueurs
- **Industrie** : Maintenance prédictive, contrôle qualité
- **Recherche** : Découverte de relations causales dans les données

---

## Références

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need*. NeurIPS.
- Martins, A. F. T., & Astudillo, R. F. (2016). *From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification*. ICML.
- Isomura, T., Shimizu, R., & Goto, M. (2023). *Optimizing FT-Transformer: Sparse Attention for Improved Performance and Interpretability*.
- Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). *Revisiting Deep Learning Models for Tabular Data*.
- Devlin, J., et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.

---

<div align="center">

**FTT+ : Feature Tokenizer Transformers interprétables**

*Développé par Léonel VODOUNOU*  
*2025*

[![Documentation](https://img.shields.io/badge/Documentation-Available-blue)](docs-url)
[![GitHub](https://img.shields.io/badge/Source-GitHub-black)](repository-url)

</div>