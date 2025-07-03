# FT-Transformer pour la Prédiction de Churn Client

Ce projet implémente plusieurs variantes de Feature Tokenizer Transformer (FT-Transformer) pour la prédiction de churn client, avec un focus sur l'interprétabilité et l'optimisation des performances.

## 🏗️ Architecture des Modèles

### **FTT (Standard)**
- **Description** : Implémentation de base du FT-Transformer
- **Caractéristiques** : Architecture Transformer classique avec tokenisation des features
- **Usage** : Modèle de référence pour comparaisons

### **FTT+ (Interprétable)**
- **Description** : Extension interprétable du FT-Transformer avec mécanismes d'attention analysables
- **Caractéristiques** :
  - Token CLS pour l'importance des features
  - Extraction des matrices d'attention
  - Heatmaps d'interactions complètes
- **Usage** : Analyse d'interprétabilité et sélection de features

### **FTT++ (Sparse + Optimisé)**
- **Description** : Pipeline en deux étapes combinant FTT+ et modèle Random avec attention sparse
- **Caractéristiques** :
  - **Étape 1** : Entraînement FTT+ → Sélection des M features les plus importantes
  - **Étape 2** : Modèle Random avec k interactions aléatoires sur features sélectionnées
  - Attention sparse pour réduction de complexité
- **Usage** : Modèle optimisé pour production avec interprétabilité maintenue

## 📊 Dataset

**Telecom Customer Churn** : Dataset de télécommunications avec 19 features (3 numériques, 16 catégorielles) pour prédire le churn client.

**Features** :
- Numériques : `tenure`, `MonthlyCharges`, `TotalCharges`
- Catégorielles : `gender`, `SeniorCitizen`, `Partner`, `Contract`, `PaymentMethod`, etc.

## 🚀 Utilisation

### **Entraînement FTT**
```bash
python train/Telecom/train_ftt/train.py
```

### **Entraînement FTT+**
```bash
python train/Telecom/train_ftt_plus/train.py
```

### **Entraînement FTT++**
```bash
# Configuration de base
python train/Telecom/train_ftt_plus_plus/train.py

# Configuration personnalisée
python train/Telecom/train_ftt_plus_plus/train.py \
    --embedding_type "Q-LR" \
    --M 12 \
    --k 6 \
    --stage1_epochs 75 \
    --stage2_epochs 50 \
    --lr 0.0005 \
    --seed 42
```

### **Depuis Kaggle**
```python
import os
os.chdir('/kaggle/working/customer-churn-ft_transformer')

!PYTHONPATH=/kaggle/working/customer-churn-ft_transformer \
    python train/Telecom/train_ftt_plus_plus/train.py \
    --embedding_type "Q-LR" --M 15 --k 8
```

## ⚙️ Paramètres

### **Paramètres FTT++**
| Paramètre | Description | Défaut |
|-----------|-------------|---------|
| `--M` | Nombre de features à sélectionner | 10 |
| `--k` | Nombre d'interactions aléatoires | 5 |
| `--embedding_type` | Type d'embedding numérique | "LR" |
| `--stage1_epochs` | Époques pour FTT+ | 50 |
| `--stage2_epochs` | Époques pour Random | 50 |
| `--lr` | Taux d'apprentissage | 1e-3 |
| `--d_token` | Dimension des tokens | 64 |
| `--n_blocks` | Nombre de blocs Transformer | 2 |

### **Types d'Embedding**
- `"LR"` : Linear + ReLU (défaut)
- `"Q-LR"` : Quantile + Linear + ReLU
- `"T"`, `"T-L"`, `"T-LR"`, `"T-LR-LR"` : Embeddings supervisés

## 📈 Résultats et Visualisations

### **Métriques Générées**
- ROC-AUC, PR-AUC, Accuracy, F1-Score
- Matthews Correlation Coefficient (MCC)
- Sensitivity, Specificity, Cohen's Kappa

### **Visualisations Automatiques**
- **Graphiques d'importance** : Features les plus influentes
- **Heatmaps d'attention** : Interactions feature-to-feature complètes
- **Comparaisons FTT+ vs Random** : Évolution de l'importance des features

### **Fichiers Sauvegardés**
```
results/results_telecom/
├── métriques/
│   ├── *_model_performance_metrics_seed_*.json
│   └── *_feature_importance_analysis_seed_*.json
├── heatmaps/
│   ├── *_feature_importance_chart_seed_*.png
│   └── *_attention_heatmap_seed_*.png
└── best_models/
    └── *_trained_model_weights_seed_*.pt
```

## 🔧 Structure du Projet

```
.
├── ftt_plus/                    # FTT+ Interprétable
│   ├── model.py                 # Architecture du modèle
│   ├── attention.py             # Mécanismes d'attention
│   └── visualisation.py         # Visualisations FTT+
├── ftt_plus_plus/               # FTT++ Pipeline
│   ├── pipeline.py              # Orchestration complète
│   ├── training_stages.py       # Étapes d'entraînement
│   ├── random_model.py          # Modèle Random avec attention sparse
│   ├── config.py                # Configurations
│   └── visualisation.py         # Visualisations FTT++
├── train/Telecom/               # Scripts d'entraînement
│   ├── train_ftt/
│   ├── train_ftt_plus/
│   └── train_ftt_plus_plus/
├── data/                        # Traitement des données
├── rtdl_lib/                    # Bibliothèque RTDL
└── interpretability_analyzer.py # Analyseur d'interprétabilité générique
```

## 🎯 Pipeline FTT++ Détaillé

### **Étape 1 : Entraînement FTT+**
1. Entraînement complet du modèle FTT+ sur toutes les features
2. Analyse d'interprétabilité avec extraction des scores d'importance
3. Sélection des M features les plus importantes
4. Génération des visualisations (graphiques + heatmaps)

### **Étape 2 : Modèle Random**
1. Création du modèle Random sur les M features sélectionnées
2. Génération de k interactions feature-feature aléatoires
3. Entraînement avec attention sparse
4. Analyse comparative des performances et interprétabilité

### **Avantages FTT++**
- **Réduction de complexité** : Attention sparse sur features sélectionnées
- **Maintien de l'interprétabilité** : Heatmaps et importance des features
- **Performances optimisées** : Pipeline en deux étapes
- **Reproductibilité** : Gestion des seeds et configurations

## 📋 Prérequis

```bash
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install scikit-learn
pip install zero  # Pour les data loaders
```

## 🔬 Analyse d'Interprétabilité

Le système d'analyse d'interprétabilité est automatiquement déclenché et génère :

1. **Scores d'importance CLS → Features**
2. **Matrices d'attention complètes**
3. **Heatmaps d'interactions**
4. **Comparaisons entre modèles**
5. **Statistiques de sparsité**

L'analyseur supporte tous les modèles FTT et adapte automatiquement les visualisations selon le type de modèle.

## 📝 Citation

Ce projet implémente et étend les concepts du FT-Transformer pour l'interprétabilité et l'optimisation des performances sur des données tabulaires de churn client.