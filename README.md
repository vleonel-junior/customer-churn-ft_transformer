# FTT+ & FTT++ : Interprétabilité avancée des Transformers pour données tabulaires

---

## Présentation

Ce dépôt propose une étude complète et des implémentations professionnelles de deux architectures innovantes pour l’apprentissage sur données tabulaires : **FTT+** (FT-Transformer Plus) et **FTT++** (FT-Transformer Plus Plus), inspirées des travaux de Tokimasa Isomura et al. L’objectif est de concilier **performance** et **interprétabilité**, deux enjeux majeurs pour l’IA appliquée aux données structurées.

---

## 1. FTT+ : Transformer interprétable pour données tabulaires

### Principe

FTT+ adapte le mécanisme des Transformers (issus du NLP) aux spécificités des données tabulaires, en introduisant :

- **Tokenisation des features** : chaque variable (numérique ou catégorielle) est encodée en vecteur dense via un `FeatureTokenizer`, produisant une séquence uniforme de tokens.
- **Ajout du token CLS** : un vecteur spécial, appris, est concaténé en tête de séquence. Il sert de point de collecte de l’information globale, à la manière de BERT.
- **Attention sélective et parcimonieuse** : l’attention n’est calculée qu’entre le token CLS et les features (dans les deux sens), excluant les interactions feature↔feature et l’auto-attention. Cela limite le surapprentissage et cible les relations vraiment utiles.
- **Partage de la matrice Value (V) entre toutes les têtes** : innovation clé pour garantir que la moyenne des scores d’attention reflète directement l’importance réelle de chaque feature.

![Schéma du partage de la matrice Value (V) entre toutes les têtes](Interpretable%20Multi-Head%20Attention.png)
*Schéma du partage de la matrice Value (V) entre toutes les têtes, garantissant l’interprétabilité des scores d’attention.*

- **Moyenne des matrices d’attention** : la matrice d’attention finale, moyennée sur les têtes, est exploitée pour l’interprétabilité (importance des features, visualisations…).

### Pipeline d’un bloc FTT+

![Vue d’ensemble d’un bloc Transformer adapté aux données tabulaires (FTT+)](One%20Transformer%20layer.png)
*Vue d’ensemble d’un bloc Transformer adapté aux données tabulaires (FTT+).*


1. **Tokenisation** des features + ajout du token CLS.
2. **Projection Q/K/V** : Q et K spécifiques à chaque tête, V partagée.
3. **Calcul des scores d’attention** (scaled dot-product, normalisé par √d_head).

![Illustration du calcul d’attention par produit scalaire (scaled dot-product attention)](Scaled%20Dot-Product%20Attention.png)
*Illustration du calcul d’attention par produit scalaire (scaled dot-product attention), cœur du mécanisme Transformer.*

4. **Application du masque** : seules les interactions CLS↔features sont autorisées.
5. **Softmax** sur les scores masqués → poids d’attention.
6. **Somme pondérée** des valeurs V selon les poids d’attention.
7. **Connexion résiduelle, normalisation, Feed-Forward, skip connection**.
8. **Sortie** : représentation enrichie de chaque token + matrice d’attention interprétable.

### Intérêt

- **Interprétabilité directe** : importance des features accessible via la matrice d’attention.
- **Réduction du surapprentissage** : attention parcimonieuse adaptée aux données tabulaires.
- **Performance** : architecture robuste, inspirée de RTDL, adaptée à la nature des données structurées.

---

## 2. FTT++ : Sélection de features et attention randomisée

### Principe

FTT++ va plus loin en combinant :

1. **Étape 1 : Entraînement d’un FTT+**
   - On entraîne un modèle FTT+ sur l’ensemble des données.
   - On extrait les scores d’importance des features via la matrice d’attention CLS↔features.
   - On sélectionne les M features les plus importantes (M = hyperparamètre).

2. **Étape 2 : Entraînement d’un modèle Random sparse**
   - On entraîne un modèle à attention randomisée sur les M features sélectionnées.
   - L’attention est calculée :
     - Entre le token CLS et chaque feature sélectionnée (comme FTT+)
     - Pour k paires de features choisies aléatoirement (k = hyperparamètre)
     - L’auto-attention reste interdite.

### Intérêt

- **Focalisation sur les variables clés** : la sélection de features maximise la pertinence de l’attention.
- **Simplicité et robustesse** : l’attention randomisée limite la complexité tout en explorant des interactions internes.
- **Interprétabilité accrue** : chaque étape fournit des scores d’importance exploitables pour l’analyse.

---

## 3. Structure du dépôt

```
ftt_plus/
    attention.py         # Mécanismes d'attention sélective et interprétable
    model.py             # Architecture FTT+ complète (tokenizer, CLS, blocs, head)
    visualisation.py     # Outils de visualisation (barplots, heatmaps)

ftt_plus_plus/
    config/              # Configurations et mapping des features
    core/                # Modèles FTT+, random sparse, attention
    training/            # Scripts d'entraînement pour chaque étape
    pipeline/            # Orchestration complète FTT++
    visualisation/       # Visualisations avancées FTT++
    __init__.py          # Import centralisé des composants
```

---

## 4. Visualisation & Interprétabilité

- **Barplots d’importance** : importance des features selon l’attention CLS.
- **Heatmaps d’interactions** : matrice d’attention complète pour analyse fine.
- **Export des scores** : pour reporting, audit, ou intégration métier.

---

## 5. Pourquoi cette étude ?

- **Comprendre et expliquer les décisions des modèles tabulaires** : enjeu crucial en entreprise (banque, assurance, santé…).
- **Allier performance et transparence** : lever le « black box effect » des réseaux profonds.
- **Proposer des outils réutilisables et adaptables** : code modulaire, visualisations prêtes à l’emploi.

---

## 6. Références

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need*. NeurIPS.
- Isomura, T., Shimizu, R., & Goto, M. (2023). *Optimizing FT-Transformer: Sparse Attention for Improved Performance and Interpretability*.
- Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). *Revisiting Deep Learning Models for Tabular Data*.
- Devlin, J., et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.

---

## 7. Auteur

Léonel VODOUNOU  
FTT+ / FTT++ – Interprétabilité avancée pour données tabulaires  
2025

---
