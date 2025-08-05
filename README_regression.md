# Sparse FTT+ pour Régression

Ce module fournit une implémentation du modèle **Sparse FTT+ (Feature Tokenizer Transformer Plus)** spécialement adaptée pour les tâches de régression sur données tabulaires.

## Caractéristiques principales

- **Architecture Transformer** optimisée pour les données tabulaires
- **Attention sparse** avec sparsemax pour une meilleure interprétabilité
- **Mécanisme d'attention interprétable** permettant l'analyse de l'importance des features
- **Support multi-types** : features numériques continues et catégorielles
- **Fonctions de perte multiples** : MSE, MAE, Huber Loss, Log-Cosh Loss
- **Métriques complètes** : MSE, RMSE, MAE, R², MAPE
- **Visualisations** intégrées pour l'analyse des résultats

## Structure des fichiers

```
sparse_ftt_plus_regression/
├── __init__.py                 # Module principal
├── model.py                    # Modèle InterpretableFTTPlusRegression

utils_regression.py             # Utilitaires pour la régression
train_regression_example.py     # Script d'entraînement d'exemple
test_sparse_ftt_regression.py   # Tests de validation
README_regression.md            # Cette documentation
```

## Installation et dépendances

Assurez-vous d'avoir les dépendances suivantes installées :

```bash
pip install torch torchvision
pip install numpy pandas scikit-learn
pip install matplotlib seaborn
pip install optuna  # Pour l'optimisation d'hyperparamètres (optionnel)
```

Le module utilise également :
- `sparse_ftt_plus.attention` (module d'attention sparse existant)
- `rtdl_lib` (bibliothèque RTDL)

## Utilisation rapide

### 1. Création d'un modèle

```python
from sparse_ftt_plus_regression.model import InterpretableFTTPlusRegression

# Configuration du modèle
model = InterpretableFTTPlusRegression.make_baseline(
    n_num_features=5,           # Nombre de features numériques
    cat_cardinalities=[3, 4, 2], # Cardinalités des features catégorielles
    d_token=128,                # Dimension des tokens
    n_blocks=3,                 # Nombre de blocs Transformer
    n_heads=8,                  # Nombre de têtes d'attention
    attention_dropout=0.1,      # Dropout de l'attention
    ffn_d_hidden=256,           # Taille cachée du FFN
    ffn_dropout=0.1,            # Dropout du FFN
    residual_dropout=0.0,       # Dropout résiduel
    d_out=1                     # Dimension de sortie (1 pour régression univariée)
)
```

### 2. Forward pass

```python
import torch

# Données d'exemple
batch_size = 32
X_num = torch.randn(batch_size, 5)      # Features numériques
X_cat = torch.randint(0, 3, (batch_size, 3))  # Features catégorielles

# Prédiction simple
predictions = model(X_num, X_cat)

# Prédiction avec extraction d'attention
predictions, attention = model(X_num, X_cat, return_attention=True)
```

### 3. Entraînement

```python
from utils_regression import get_regression_loss_function

# Configuration
optimizer = torch.optim.AdamW(model.optimization_param_groups(), lr=1e-4)
loss_fn = get_regression_loss_function('mse')  # 'mse', 'mae', 'huber', 'logcosh'

# Boucle d'entraînement
model.train()
for epoch in range(num_epochs):
    for X_num_batch, X_cat_batch, y_batch in dataloader:
        optimizer.zero_grad()
        predictions = model(X_num_batch, X_cat_batch)
        loss = loss_fn(predictions, y_batch)
        loss.backward()
        optimizer.step()
```

### 4. Évaluation

```python
from utils_regression import regression_performance_dict, print_regression_metrics

# Prédictions
model.eval()
with torch.no_grad():
    predictions = model(X_num_test, X_cat_test)

# Calcul des métriques
metrics = regression_performance_dict(y_true.numpy(), predictions.numpy())
print_regression_metrics(metrics)
```

### 5. Analyse d'interprétabilité

```python
# Importance des features basée sur l'attention CLS
feature_names = ['feature_1', 'feature_2', 'feature_3', ...]
importance = model.get_cls_importance(X_num, X_cat, feature_names=feature_names)

print("Importance des features:")
for name, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {score:.4f}")
```

## Script d'exemple complet

Utilisez le script `train_regression_example.py` pour un exemple complet :

```bash
python train_regression_example.py
```

Ce script :
- Crée un dataset synthétique de régression
- Divise les données en train/validation/test
- Entraîne le modèle avec early stopping
- Évalue les performances avec toutes les métriques
- Analyse l'importance des features
- Sauvegarde les résultats

## Tests

Validez l'installation avec le script de test :

```bash
python test_sparse_ftt_regression.py
```

Ce script teste :
- Création du modèle
- Forward pass
- Fonctions de perte
- Calcul des métriques
- Importance des features
- Groupes de paramètres d'optimisation
- Utilitaires de données

## Configuration recommandée

Pour de bonnes performances, utilisez ces paramètres comme point de départ :

```python
# Configuration baseline
config = {
    'd_token': 128,           # 64, 128, 256
    'n_blocks': 3,            # 2-6 blocs
    'n_heads': 8,             # 4, 8, 16
    'attention_dropout': 0.1, # 0.0-0.3
    'ffn_d_hidden': 256,      # 2-4x d_token
    'ffn_dropout': 0.1,       # 0.0-0.3
    'residual_dropout': 0.0,  # 0.0-0.2
    'attention_mode': 'hybrid' # 'sparse', 'full', 'hybrid'
}

# Entraînement
training_config = {
    'lr': 1e-4,               # 1e-5 à 1e-3
    'weight_decay': 1e-5,     # 1e-6 à 1e-3
    'batch_size': 32,         # 16, 32, 64, 128
    'loss_type': 'mse',       # 'mse', 'mae', 'huber'
    'normalize_targets': True, # Recommandé pour la régression
    'patience': 15            # Early stopping
}
```

## Fonctions de perte disponibles

- **MSE** (`'mse'`) : Mean Squared Error (par défaut)
- **MAE** (`'mae'`) : Mean Absolute Error (robuste aux outliers)
- **Huber** (`'huber'`) : Compromis entre MSE et MAE
- **Log-Cosh** (`'logcosh'`) : Approximation lisse de MAE

## Métriques d'évaluation

Le module calcule automatiquement :
- **MSE** : Mean Squared Error
- **RMSE** : Root Mean Squared Error
- **MAE** : Mean Absolute Error  
- **R²** : Coefficient de détermination
- **MAPE** : Mean Absolute Percentage Error

## Normalisation des cibles

Pour améliorer la convergence, il est recommandé de normaliser les cibles :

```python
from utils_regression import normalize_targets, denormalize_predictions

# Normalisation
y_train_norm, y_mean, y_std = normalize_targets(y_train)

# Après prédiction, dénormaliser
predictions_denorm = denormalize_predictions(predictions_norm, y_mean, y_std)
```

## Modes d'attention

Le modèle supporte différents modes d'attention :

- **`'hybrid'`** : Combinaison d'attention dense et sparse (recommandé)
- **`'sparse'`** : Attention entièrement sparse avec sparsemax
- **`'full'`** : Attention dense classique

## Optimisation des hyperparamètres

Pour optimiser les hyperparamètres, vous pouvez adapter le code d'Optuna existant en changeant :

1. La fonction de perte (MSE au lieu de BCEWithLogitsLoss)
2. Les métriques d'évaluation (R² au lieu d'AUC)
3. Le modèle (InterpretableFTTPlusRegression)

## Limitations

- Le modèle est conçu pour la régression univariée (d_out=1)
- Pour la régression multivariée, ajustez `d_out` et adaptez les métriques
- Les features catégorielles doivent être encodées en entiers (0, 1, 2, ...)
- La normalisation des features numériques est recommandée

## Performance

Le modèle Sparse FTT+ pour régression offre :
- **Bonne performance** sur données tabulaires hétérogènes
- **Interprétabilité** grâce au mécanisme d'attention
- **Robustesse** aux different types de features
- **Flexibilité** dans le choix des fonctions de perte

## Support

Pour des questions ou des problèmes, vérifiez d'abord avec le script de test :
```bash
python test_sparse_ftt_regression.py
```

Cela validera que toutes les composantes fonctionnent correctement.