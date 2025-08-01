# FTT Utils

Ce module fournit des utilitaires supplémentaires pour le FT-Transformer standard (FTT) de `rtdl_lib`.

## Fonctions

### `make_baseline_with_n_heads`

```python
def make_baseline_with_n_heads(
    *,
    n_num_features: int,
    cat_cardinalities: Optional[List[int]],
    d_token: int,
    n_blocks: int,
    attention_n_heads: int,
    attention_dropout: float,
    ffn_d_hidden: int,
    ffn_dropout: float,
    residual_dropout: float,
    last_layer_query_idx: Union[None, List[int], slice] = None,
    kv_compression_ratio: Optional[float] = None,
    kv_compression_sharing: Optional[str] = None,
    d_out: int,
) -> 'FTTransformer':
```

Cette fonction est une extension de `FTTransformer.make_baseline()` qui permet de spécifier le nombre de têtes d'attention.

#### Args:
- `n_num_features`: le nombre de features numériques continues
- `cat_cardinalities`: les cardinalités des features catégorielles
- `d_token`: la taille des tokens. Doit être un multiple de `attention_n_heads`.
- `n_blocks`: le nombre de blocs Transformer
- `attention_n_heads`: le nombre de têtes d'attention
- `attention_dropout`: le dropout pour les blocs d'attention
- `ffn_d_hidden`: la taille d'entrée pour la deuxième couche linéaire dans `Transformer.FFN`.
- `ffn_dropout`: le taux de dropout après la première couche linéaire dans `Transformer.FFN`.
- `residual_dropout`: le taux de dropout pour la sortie de chaque branche résiduelle de tous les blocs Transformer.
- `last_layer_query_idx`: indices des tokens qui doivent être traités par le dernier bloc Transformer.
- `kv_compression_ratio`: applique la technique de [wang2020linformer] pour accélérer les modules d'attention quand le nombre de features est grand.
- `kv_compression_sharing`: politique de partage des poids pour `kv_compression_ratio`.
- `d_out`: la taille de sortie.

#### Returns:
- `FTTransformer`: modèle configuré

#### Exemple:
```python
from ftt_utils import make_baseline_with_n_heads

model = make_baseline_with_n_heads(
    n_num_features=3,
    cat_cardinalities=[2, 3],
    d_token=128,
    n_blocks=3,
    attention_n_heads=16,
    attention_dropout=0.2,
    ffn_d_hidden=256,
    ffn_dropout=0.2,
    residual_dropout=0.0,
    d_out=1,
)