# FTT++ (FT-Transformer Plus Plus) - Implémentation Complète

## 🎯 Vue d'Ensemble

FTT++ est une approche en deux étapes qui améliore l'interprétabilité de FTT+ tout en maintenant de hautes performances. Cette implémentation suit fidèlement la méthodologie proposée par Isomura et al. avec **entraînement intégré**.

### 🔬 Principe Fondamental

FTT++ combine la **focalisation intelligente** de FTT+ avec une **exploration contrôlée** des interactions feature-feature, créant un modèle à la fois performant et hautement interprétable.

## 📋 Architecture du Projet

```
ftt_plus_plus/
├── __init__.py                 # Module principal FTT++
├── sparse_attention.py         # Mécanisme d'attention sparse
├── random_model.py            # Étape 2: Modèle Random avec attention sparse
└── pipeline.py                # Orchestration complète avec entraînement intégré

train/Telecom/train_ftt_plus_plus/
└── train.py                   # Script d'entraînement pour dataset Telecom
```

## 🚀 Workflow FTT++ en Deux Étapes Intégrées

### **Étape 1 : Entraînement FTT+ et Sélection de Features**

```python
# 1. Entraîner un modèle FTT+ complet avec early stopping
model_ftt_plus = InterpretableFTTPlus.make_baseline(...)
for epoch in range(n_epochs):
    train_loss = train(epoch, model_ftt_plus, optimizer, X, y, train_loader, loss_fn)
    val_loss = val(epoch, model_ftt_plus, X, y, val_loader, loss_fn)

# 2. Analyser l'importance avec interpretability_analyzer
interpretability_results = analyze_interpretability(
    model=model_ftt_plus, X=X, y=y, model_name='interpretable_ftt_plus',
    seed=seed, model_config=model_config, ...
)

# 3. Sélectionner les M features les plus importantes
cls_importance = interpretability_results['cls_importance']
selected_features = [name for name, score in sorted(cls_importance.items(), 
                    key=lambda x: x[1], reverse=True)[:M]]
```

### **Étape 2 : Entraînement Random avec Attention Sparse**

```python
# 1. Créer un modèle Random focalisé sur les M features sélectionnées
model_random = InterpretableFTTRandom.from_selected_features(
    selected_feature_indices_num=indices_num,
    selected_feature_indices_cat=indices_cat,
    k=5  # Nombre d'interactions feature-feature aléatoires
)

# 2. Entraîner le modèle Random avec attention sparse
for epoch in range(n_epochs):
    train_loss = train(epoch, model_random, optimizer, X, y, train_loader, loss_fn)
    val_loss = val(epoch, model_random, X, y, val_loader, loss_fn)
```

## 🧠 Innovation Technique

### **Attention Sparse Contrôlée**

L'attention dans le modèle Random suit un pattern sparse spécifique :

```python
# Masque d'attention sparse
mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)

# 1. CLS ↔ toutes les features sélectionnées
mask[0, 1:] = True  # CLS vers features
mask[1:, 0] = True  # Features vers CLS

# 2. k paires d'interactions feature-feature aléatoires
for i, j in random_pairs:
    mask[i, j] = True
    mask[j, i] = True

# 3. Auto-attention interdite (diagonale reste False)
```

### **Intégration avec interpretability_analyzer.py**

- **Utilisation native** du module d'analyse existant
- **Sauvegarde automatique** dans `results/results_telecom/`
- **Graphiques d'importance** générés automatiquement
- **Pas de duplication** de code d'analyse

## 📊 Utilisation

### **Entraînement Complet Intégré**

```bash
# Entraînement FTT++ avec 10 features sélectionnées et 5 interactions aléatoires
cd train/Telecom/train_ftt_plus_plus
python train.py --M 10 --k 5 --seed 0

# Avec configuration personnalisée
python train.py \
    --M 15 \
    --k 8 \
    --stage1_epochs 100 \
    --stage2_epochs 50 \
    --d_token 128 \
    --ffn_hidden 256 \
    --embedding_type Q-LR
```

### **Pipeline Programmatique**

```python
from ftt_plus_plus.pipeline import FTTPlusPlusPipeline

# Configuration
pipeline = FTTPlusPlusPipeline(M=10, k=5, attention_seed=42)

# Exécution complète avec entraînements intégrés
results = pipeline.run_complete_pipeline(
    X=X, y=y, 
    cat_cardinalities=cat_cardinalities,
    feature_names=feature_names,
    stage1_epochs=50,
    stage2_epochs=50,
    embedding_type="LR",
    device=device
)

# Analyse des résultats
print(f"Features sélectionnées: {results['selected_features']}")
print(f"Sparsité atteinte: {results['comparison']['sparsity_achieved']:.2%}")
```

## 📈 Avantages de FTT++

### **1. Workflow Naturel et Intégré**

- **Entraînement continu** : FTT+ → Analyse → Sélection → Random
- **Pas de modèles pré-entraînés** à charger
- **Utilisation native** d'`interpretability_analyzer.py`
- **Pipeline unifié** pour toute l'expérimentation

### **2. Interprétabilité Maximale**

- **Focalisation claire** : Identification explicite des M features les plus importantes
- **Interactions contrôlées** : Exploration limitée des relations feature-feature
- **Attention sparse** : Visualisation directe des connexions importantes
- **Analyse comparative** : Évolution de l'importance FTT+ → Random

### **3. Performance Maintenue**

- **Transfert de connaissance** : Les features importantes sont identifiées par FTT+
- **Architecture optimisée** : Focus sur les interactions vraiment utiles
- **Robustesse** : Réduction du sur-apprentissage grâce à la sparsité

### **4. Efficacité Computationnelle**

- **Moins de paramètres** : Seules les M features importantes sont traitées
- **Calculs réduits** : Attention sparse vs attention complète
- **Scalabilité** : Performance dégradée gracieusement avec la taille des données

## 🔍 Exemple de Résultats

### **Workflow d'Entraînement**

```
🚀 === PIPELINE FTT++ COMPLET ===

🚀 === ÉTAPE 1: Entraînement FTT+ Complet ===
📊 Modèle FTT+ créé avec 186,457 paramètres
⏳ Entraînement du modèle FTT+ en cours...
Epoch 023 | Train Loss: 0.4234 | Val Loss: 0.4567 | Time: 1.23s
✅ Meilleur modèle chargé (époque 23, val_loss: 0.4567)

🔍 Analyse d'interprétabilité avec interpretability_analyzer...
📋 Features sélectionnées:
  1. Contract             : 0.1234
  2. tenure               : 0.0987
  3. MonthlyCharges       : 0.0876

🎯 === ÉTAPE 2: Entraînement Modèle Random ===
📊 Modèle Random créé avec 89,123 paramètres
🔗 Sparsité d'attention: 75.21%
⏳ Entraînement du modèle Random en cours...
✅ Meilleur modèle Random chargé (époque 18, val_loss: 0.4123)
```

### **Analyse Comparative**

```
📊 === ANALYSE DÉTAILLÉE DES RÉSULTATS ===
🔍 Comparaison des modèles:
   - Paramètres FTT+: 186,457
   - Paramètres Random: 89,123
   - Réduction: 52.20%

📈 Corrélation d'importance FTT+ ↔ Random: 0.834

📋 Features sélectionnées (M=10) - Évolution de l'importance:
Rang Feature              FTT+ Score   Random Score  Évolution
1.   Contract             0.1234       0.1456        +18.0%
2.   tenure               0.0987       0.1123        +13.8%
3.   MonthlyCharges       0.0876       0.0934        +6.6%
```

### **Statistiques d'Attention Sparse**

```
🔗 Statistiques d'attention sparse:
   - Connexions totales possibles: 121
   - Connexions actives: 30
   - Sparsité: 75.21%
   - Connexions CLS ↔ features: 20
   - Connexions feature ↔ feature: 10
```

## 🛠️ Configuration Avancée

### **Paramètres Clés**

| Paramètre | Description | Valeur Recommandée |
|-----------|-------------|-------------------|
| `M` | Nombre de features sélectionnées | 10-15 (50-75% du total) |
| `k` | Interactions feature-feature | 3-8 (≈M/2) |
| `attention_seed` | Seed pour reproductibilité | 42 |
| `d_token` | Dimension des tokens | 64-128 |
| `n_blocks` | Blocs Transformer | 2-4 |
| `embedding_type` | Type d'embedding numérique | LR, Q-LR, T-LR |

### **Types d'Embeddings Supportés**

- **Linear** : L, LR, LR-LR
- **Quantile** : Q, Q-L, Q-LR, Q-LR-LR
- **Tree-based** : T, T-L, T-LR, T-LR-LR
- **Periodic** : P, P-L, P-LR, P-LR-LR

## 📊 Analyse et Sauvegarde

### **Sauvegarde Structurée**

```
results/results_telecom/
├── métriques/
│   ├── interpretable_ftt_plus_model_performance_metrics_seed_0.json
│   ├── interpretable_ftt_plus_feature_importance_analysis_seed_0.json
│   └── ftt_plus_plus_complete_results.json
├── heatmaps/
│   └── interpretable_ftt_plus_feature_importance_chart_seed_0.png
└── best_models/
    └── interpretable_ftt_plus_trained_model_weights_seed_0.pt
```

### **Résultats Locaux**

```
outputs/ftt_plus_plus_seed_0_M_10_k_5/
└── ftt_plus_plus_results.json
```

## 🎯 Cas d'Usage Recommandés

### **1. Interprétabilité Critique**
- Applications médicales
- Finance et crédit
- Analyse de risque

### **2. Optimisation de Performance**
- Datasets avec beaucoup de features
- Contraintes computationnelles
- Déploiement en production

### **3. Recherche et Analyse**
- Découverte de relations feature-feature
- Validation de l'importance des variables
- Comparaison de modèles

## 🔧 Architecture Simplifiée

```python
# Architecture finale simplifiée
ftt_plus_plus/
├── sparse_attention.py     # Mécanisme d'attention sparse
├── random_model.py        # Modèle Random avec features sélectionnées
└── pipeline.py            # Pipeline intégré (entraîne FTT+ + analyse + Random)

# Plus besoin de:
# ❌ feature_selector.py    # Remplacé par interpretability_analyzer.py
# ❌ Modèles pré-entraînés  # Entraînement intégré dans le pipeline
```

## 🚀 Prochaines Étapes

1. **Entraînement sur vos données** : Utiliser le pipeline intégré
2. **Optimisation des hyperparamètres** : Tuner M, k selon vos besoins
3. **Analyse comparative** : Comparer avec FTT+ standard et autres baselines
4. **Déploiement** : Intégrer le modèle Random optimisé en production

---

**FTT++** avec entraînement intégré représente une approche naturelle et efficace pour l'équilibre performance-interprétabilité sur données tabulaires. Le pipeline unifié simplifie l'expérimentation tout en maximisant la réutilisation du code existant.