# FTT+ : Feature Tokenizer Transformers interprétables pour données tabulaires

---

## 1. Introduction

Ce dépôt propose une architecture pour l’apprentissage sur données tabulaires :  
- **FTT+** (FT-Transformer Plus) : attention sélective et interprétable.

L’objectif : concilier **performance** et **interprétabilité** sur des données structurées.

---

## 2. FTT+ : Forward Pass et Composants

> **Nouveauté : FTT+ supporte plusieurs schémas d'attention :**
> - `cls` : attention uniquement entre le token CLS et les features (FTT+ original)
> - `hybrid` : attention CLS↔features et features↔features (hors diagonale, par défaut)
> - `full` : attention complète entre toutes les positions (hors diagonale)
>
> Le mode par défaut est **`hybrid`** (interactions CLS↔features et features↔features).

### Schéma global du forward pass

<p align="center">
  <img src="images/FT_Transformer architecture.png" alt="Architecture globale du FT-Transformer appliqué aux données tabulaires" width="750"/>
</p>
<p align="center"><b>Architecture globale du FT-Transformer appliqué aux données tabulaires</b></p>

1. **Tokenisation des features**  
   - `FeatureTokenizer` encode chaque variable (numérique/catégorielle) en vecteur dense.
   ![Illustration du processus de tokenisation des variables brutes en vecteurs denses](images/Illustration%20d'un%20Feature%20Tokenizer.png)

   *Illustration du processus de tokenisation des variables brutes en vecteurs denses.*

2. **Ajout du token CLS**  
   - Un vecteur spécial, appris, est ajouté en tête de séquence.

3. **Passage dans les blocs Transformer**  
   - Chaque bloc applique :
     - **Interpretable Multi-Head Attention** :  
       - Q/K spécifiques à chaque tête, V partagée.
       - **Schéma d'attention flexible** :  
         - Par défaut, interactions CLS↔features **et** features↔features (hors diagonale).
         - Possibilité de forcer le mode `cls` (FTT+ original) ou `full` via la configuration.
       - Moyenne des scores d’attention sur les têtes pour interprétabilité directe.

       <p align="center">
         <img src="images/Scaled Dot-Product Attention.png" alt="Scaled Dot-Product Attention adapté FTT+ (CLS↔features uniquement)" width="500"/>
       </p>
       <p align="center"><b>Scaled Dot-Product Attention : les interactions autorisées dépendent du mode choisi (`cls`, `hybrid`, `full`).</b></p>

       <br>

       <p align="center">
         <img src="images/Interpretable Multi-Head Attention.png" alt="Illustration de l'Interpretable Multi-Head Attention" width="500"/>
       </p>
       <p align="center"><b>Interpretable Multi-Head Attention : la moyenne des scores d’attention reflète l’importance réelle de chaque feature.</b></p>

     - **Feed-Forward Network (FFN)** :  
       - Transformation non-linéaire classique.
     - **Normalisation & Résidualité** :  
       - LayerNorm, skip connections.

   <p align="center">
     <img src="images/One Transformer layer.png" alt="Vue d’ensemble d’un bloc Transformer adapté aux données tabulaires (FTT+)" width="500"/>
   </p>
   <p align="center"><b>Vue d’ensemble d’un bloc Transformer adapté aux données tabulaires (FTT+)</b></p>

4. **Head de classification**  
   - Prédiction à partir du token CLS.

### Extraction de l’interprétabilité

- Importance des features : extraite directement de la matrice d’attention CLS→features (ou interactions selon le mode).
- Visualisations : barplots, heatmaps.

---

## 3. Structure du code

```
ftt_plus/
    attention.py         # Attention sélective/interprétable
    model.py             # Architecture FTT+ (tokenizer, CLS, blocs, head)
    visualisation.py     # Visualisation (barplots, heatmaps)
```

---

## 4. Notes d'utilisation

- Pour changer le mode d'attention de FTT+, ajoutez dans la config :
  ```python
  ftt_plus_config = {
      # ...autres paramètres...
      'attention_mode': 'cls',  # ou 'hybrid', ou 'full'
  }
  ```

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
FTT+ – Interprétabilité avancée pour données tabulaires  
2025

---
---
