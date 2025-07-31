# FTT+ : Feature Tokenizer Transformers interprétables pour données tabulaires

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Transformers-orange)](https://pytorch.org)

*Concilier **performance** et **interprétabilité** sur données tabulaires*

</div>

---

## 📋 Table des matières

- [Introduction](#-introduction)
- [Architecture FTT+](#-architecture-ftt)
- [Sparse FTT+](#-sparse-ftt)
- [Structure du code](#-structure-du-code)
- [Configuration](#-configuration)
- [Installation](#-installation)
- [Références](#-références)

---

## 🎯 Introduction

Ce dépôt propose une architecture innovante pour l'apprentissage sur données tabulaires :

- **FTT+** (FT-Transformer Plus) : attention sélective et interprétable
- **Sparse FTT+** : variante utilisant une attention sparse pour une interprétabilité encore plus fine

### 🎨 Modes d'attention supportés

| Mode | Description | Interactions |
|------|-------------|--------------|
| `cls` | FTT+ original | CLS ↔ Features uniquement |
| `hybrid` | **Mode par défaut** | CLS ↔ Features + Features ↔ Features |
| `full` | Attention complète | Toutes positions (hors diagonale) |

---

## 🏗️ Architecture FTT+

### Vue d'ensemble

<div align="center">
<img src="images/FT_Transformer architecture.png" alt="Architecture globale" width="80%" style="max-width: 700px;">
<br><em>Architecture globale du FT-Transformer pour données tabulaires</em>
</div>

### 🔄 Pipeline de traitement

#### 1. **Tokenisation des features**

<div align="center">
<img src="images/Illustration%20d'un%20Feature%20Tokenizer.png" alt="Feature Tokenizer" width="80%" style="max-width: 700px;">
<br><em>Processus de tokenisation : variables brutes → vecteurs denses</em>
</div>

Le `FeatureTokenizer` transforme chaque variable (numérique/catégorielle) en représentation vectorielle dense.

#### 2. **Token CLS & Blocs Transformer**

Chaque bloc Transformer applique séquentiellement :

<div align="center">
<img src="images/One Transformer layer.png" alt="Bloc Transformer" width="60%" style="max-width: 400px;">
<br><em>Architecture d'un bloc Transformer FTT+</em>
</div>

##### **Interpretable Multi-Head Attention**

<div align="center">
<img src="images/Scaled Dot-Product Attention.png" alt="Attention mécanisme" width="50%" style="max-width: 350px;">
<br><em>Mécanisme d'attention adaptatif selon le mode choisi</em>
</div>

**Caractéristiques clés :**
- Q/K spécifiques par tête, V partagée
- Moyenne des scores d'attention pour interprétabilité directe
- Schéma d'attention flexible selon configuration

<div align="center">
<img src="images/Interpretable Multi-Head Attention.png" alt="Multi-Head Attention" width="80%" style="max-width: 700px;">
<br><em>Interpretable Multi-Head Attention : importance réelle des features</em>
</div>

##### **Feed-Forward Network & Normalisation**
- Transformation non-linéaire classique
- LayerNorm et connexions résiduelles

#### 3. **Classification finale**
Prédiction basée sur la représentation du token CLS.

### 📊 Extraction de l'interprétabilité

L'importance des features est extraite directement de la matrice d'attention CLS→features, permettant :
- **Visualisations intuitives** : barplots, heatmaps
- **Transparence des décisions** : identification des features influents
- **Analyse des interactions** : comprendre les relations entre variables

---

## ⚡ Sparse FTT+

### Principe

Sparse FTT+ utilise l'attention sparse (`sparsemax`) au lieu de l'attention softmax standard.

### Avantages

| Aspect | Bénéfice |
|--------|----------|
| **Interactions** | Réduction du nombre d'interactions significatives |
| **Performance** | Concentration sur les relations les plus pertinentes |
| **Interprétabilité** | Identification explicite des features les plus influents |

### Fonctionnement

La fonction `sparsemax` produit des distributions d'attention :
- **Strictement positives** sur un sous-ensemble restreint
- **Nulles** ailleurs, éliminant le bruit

---

## 📁 Structure du code

```
📦 ftt_plus/
├── 📄 attention.py         # Attention sélective/interprétable
└── 📄 model.py             # Architecture FTT+ complète

📦 sparse_ftt_plus/
├── 📄 attention.py         # Attention sparse/interprétable
└──  📄 model.py             # Architecture Sparse FTT+
```

---

## ⚙️ Configuration

### Paramétrage FTT+

```python
ftt_plus_config = {
    # Architecture
    'd_model': 256,
    'n_heads': 8,
    'n_layers': 6,
    
    # Mode d'attention
    'attention_mode': 'hybrid',  # 'cls', 'hybrid', 'full'
    
    # Autres paramètres...
}
```

### Modes d'attention détaillés

- **`cls`** : Reproduction fidèle du FTT+ original
- **`hybrid`** : **Recommandé** - Équilibre performance/interprétabilité  
- **`full`** : Maximum d'interactions, plus coûteux

---

## 🔧 Installation

### Dépendances de base
```bash
pip install torch transformers numpy pandas matplotlib seaborn
```

### Pour Sparse FTT+
```bash
pip install sparsemax
```

---

## 🎯 Pourquoi cette étude ?

### Enjeux actuels

| Domaine | Problématique | Solution FTT+ |
|---------|---------------|---------------|
| **Entreprise** | Décisions opaques en finance/santé | Interprétabilité native |
| **IA Responsable** | "Black box effect" | Transparence des modèles |
| **Recherche** | Trade-off performance/explicabilité | Architecture optimisée |

### Contributions

- **🔍 Transparence** : Mécanismes d'attention interprétables
- **📈 Performance** : Architecture optimisée pour données tabulaires  
- **🛠️ Réutilisabilité** : Code modulaire et visualisations prêtes

---

## 📚 Références

- **Vaswani, A., et al.** (2017). *Attention Is All You Need*. NeurIPS.
- **Gorishniy, Y., et al.** (2021). *Revisiting Deep Learning Models for Tabular Data*.
- **Isomura, T., et al.** (2023). *Optimizing FT-Transformer: Sparse Attention for Improved Performance*.
- **Devlin, J., et al.** (2018). *BERT: Pre-training of Deep Bidirectional Transformers*.

---

<div align="center">

**🚀 FTT+ – Interprétabilité avancée pour données tabulaires**

*Développé par **Léonel VODOUNOU** • 2025*

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](votre-repo-url)
[![Documentation](https://img.shields.io/badge/Docs-Available-blue)](docs-url)

</div>
