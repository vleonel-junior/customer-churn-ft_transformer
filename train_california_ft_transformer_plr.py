"""
Script d'entraînement du modèle FT-Transformer PLR sur le dataset California Housing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
import math
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, List
import time
import sys
from pathlib import Path

# Ajouter le chemin pour importer num_embedding_factory
sys.path.append(str(Path(__file__).parent))
from num_embedding_factory import get_num_embedding

from utils_regression import (
    regression_performance_dict, print_regression_metrics, 
    normalize_targets, denormalize_predictions,
    get_regression_loss_function
)
from interpretability_analyzer import analyze_interpretability


def get_activation_fn(activation: str):
    """Retourne la fonction d'activation."""
    if activation == 'reglu':
        def reglu(x):
            a, b = x.chunk(2, dim=-1)
            return a * F.relu(b)
        return reglu
    elif activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    else:
        raise ValueError(f"Activation inconnue: {activation}")


def get_nonglu_activation_fn(activation: str):
    """Retourne la fonction d'activation non-GLU."""
    if activation.endswith('glu'):
        return F.relu
    return get_activation_fn(activation)


class TokenizerPLR(nn.Module):
    """Tokenizer FT-Transformer avec embeddings P-LR pour les variables numériques."""
    category_offsets: Optional[torch.Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: Optional[List[int]],
        d_token: int,
        bias: bool,
        X_train: Optional[torch.Tensor] = None,
        embedding_type: str = "P-LR",
    ) -> None:
        super().__init__()
        
        self.d_numerical = d_numerical
        self.d_token = d_token
        self.embedding_type = embedding_type
        
        # Embeddings catégoriels (standard)
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        # Token CLS
        self.cls_weight = nn.Parameter(torch.Tensor(1, d_token))
        nn_init.kaiming_uniform_(self.cls_weight, a=math.sqrt(5))
        
        # Embeddings numériques : utiliser P-LR si disponible, sinon standard
        if d_numerical > 0 and X_train is not None:
            print(f"Utilisation des embeddings {embedding_type} pour {d_numerical} variables numériques")
            try:
                self.num_embeddings = get_num_embedding(
                    embedding_type=embedding_type,
                    X_train=X_train.cpu() if X_train.is_cuda else X_train,
                    d_embedding=d_token,
                    y_train=None  # Pas besoin pour P-LR
                )
            except Exception as e:
                print(f"Erreur lors de la création des embeddings P-LR: {e}")
                print("Utilisation des embeddings standard")
                self.num_embeddings = None
                self.weight = nn.Parameter(torch.Tensor(d_numerical, d_token))
                nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            # Fallback : embeddings standard
            print(f"Utilisation des embeddings standard pour {d_numerical} variables numériques")
            self.num_embeddings = None
            if d_numerical > 0:
                self.weight = nn.Parameter(torch.Tensor(d_numerical, d_token))
                nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Bias
        self.bias = nn.Parameter(torch.Tensor(d_bias, d_token)) if bias else None
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return (
            1 +  # CLS token
            self.d_numerical +  # numerical features
            (0 if self.category_offsets is None else len(self.category_offsets))  # categorical features
        )

    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        
        batch_size = len(x_some)
        device = x_some.device
        
        # Token CLS
        x_cls = self.cls_weight.expand(batch_size, 1, -1)  # (batch_size, 1, d_token)
        tokens = [x_cls]
        
        # Embeddings numériques
        if x_num is not None:
            if self.num_embeddings is not None:
                # Utiliser les embeddings P-LR
                x_num_embedded = self.num_embeddings(x_num)  # (batch_size, n_features, d_token)
            else:
                # Utiliser les embeddings standard
                x_num_embedded = self.weight[None] * x_num[:, :, None]  # (batch_size, n_features, d_token)
            tokens.append(x_num_embedded)
        
        # Embeddings catégoriels
        if x_cat is not None and x_cat.shape[1] > 0 and self.category_embeddings is not None:
            x_cat_embedded = self.category_embeddings(x_cat + self.category_offsets[None])
            tokens.append(x_cat_embedded)
        
        # Concaténer tous les tokens
        x = torch.cat(tokens, dim=1)  # (batch_size, n_tokens, d_token)
        
        # Ajouter les bias
        if self.bias is not None:
            bias = torch.cat([
                torch.zeros(1, self.bias.shape[1], device=device),  # pas de bias pour CLS
                self.bias,
            ])
            x = x + bias[None]
        
        return x


class MultiheadAttention(nn.Module):
    def __init__(
        self, d: int, n_heads: int, dropout: float, initialization: str
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        key_compression: Optional[nn.Linear],
        value_compression: Optional[nn.Linear],
    ) -> torch.Tensor:
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x


class TransformerPLR(nn.Module):
    """FT-Transformer avec embeddings P-LR pour les variables numériques."""

    def __init__(
        self,
        *,
        # tokenizer
        d_numerical: int,
        categories: Optional[List[int]],
        token_bias: bool,
        embedding_type: str,
        # transformer
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_ffn_factor: float,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        activation: str,
        prenormalization: bool,
        initialization: str,
        # linformer
        kv_compression: Optional[float],
        kv_compression_sharing: Optional[str],
        #
        d_out: int,
        # Pour l'initialisation des embeddings P-LR
        X_train: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = TokenizerPLR(
            d_numerical, categories, d_token, token_bias, X_train, embedding_type
        )

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden * (2 if activation.endswith('glu') else 1)
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            self.layers.append(layer)

        self.activation = get_activation_fn(activation)
        self.last_activation = get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.tokenizer(x_num, x_cat)

        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                # for the last attention, it is enough to process only [CLS]
                (x_residual[:, :1] if is_last_layer else x_residual),
                x_residual,
                None,  # no compression
                None,  # no compression
            )
            if is_last_layer:
                x = x[:, : x_residual.shape[1]]
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        assert x.shape[1] == 1
        x = x[:, 0]
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x


class InterpretableFTTransformerPLRRegression(nn.Module):
    """Wrapper pour FT-Transformer PLR avec fonctionnalités d'interprétabilité."""
    
    def __init__(self, transformer: TransformerPLR):
        super().__init__()
        self.transformer = transformer
        
    @classmethod
    def make_baseline(
        cls,
        n_num_features: int,
        cat_cardinalities: list,
        d_token: int = 64,
        n_blocks: int = 3,
        n_heads: int = 8,
        attention_dropout: float = 0.1,
        ffn_d_hidden: int = 256,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        d_out: int = 1,
        average_layer: bool = True,
        X_train: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Crée un modèle FT-Transformer PLR baseline."""
        
        categories = None if not cat_cardinalities else cat_cardinalities
        
        transformer = TransformerPLR(
            d_numerical=n_num_features,
            categories=categories,
            token_bias=True,
            embedding_type="P-LR",
            n_layers=n_blocks,
            d_token=d_token,
            n_heads=n_heads,
            d_ffn_factor=ffn_d_hidden / d_token,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            residual_dropout=residual_dropout,
            activation='reglu',
            prenormalization=True,
            initialization='kaiming',
            kv_compression=None,
            kv_compression_sharing=None,
            d_out=d_out,
            X_train=X_train
        )
        
        return cls(transformer)
    
    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        """Forward pass."""
        return self.transformer(x_num, x_cat)
    
    def optimization_param_groups(self):
        """Retourne les groupes de paramètres pour l'optimisation."""
        def needs_wd(name):
            return all(x not in name for x in ['tokenizer', '.norm', '.bias'])
        
        parameters_with_wd = [v for k, v in self.named_parameters() if needs_wd(k)]
        parameters_without_wd = [v for k, v in self.named_parameters() if not needs_wd(k)]
        
        return [
            {'params': parameters_with_wd},
            {'params': parameters_without_wd, 'weight_decay': 0.0},
        ]
    
    def get_cls_importance(self, x_num: torch.Tensor, x_cat: torch.Tensor, feature_names: list) -> Dict[str, float]:
        """Calcule l'importance des features via le token CLS."""
        self.eval()
        with torch.no_grad():
            # Obtenir les embeddings tokenizés
            x = self.transformer.tokenizer(x_num, x_cat)  # (batch_size, n_tokens, d_token)
            
            # Forward pass jusqu'à la dernière couche pour obtenir les attentions
            for layer_idx, layer in enumerate(self.transformer.layers):
                is_last_layer = layer_idx + 1 == len(self.transformer.layers)
                
                # Attention uniquement sur CLS pour la dernière couche
                if is_last_layer:
                    x_cls = x[:, :1]  # Token CLS uniquement
                    # Calculer l'attention du CLS vers tous les tokens
                    q = layer['attention'].W_q(x_cls)
                    k = layer['attention'].W_k(x)
                    
                    # Calculer les scores d'attention
                    attention_scores = torch.matmul(q, k.transpose(-2, -1))
                    attention_scores = attention_scores / (x.shape[-1] ** 0.5)
                    attention_weights = torch.softmax(attention_scores, dim=-1)
                    
                    # Moyenner sur le batch et les têtes
                    cls_attention = attention_weights.mean(dim=0).squeeze(0)  # (n_tokens,)
                    
                    break
                else:
                    # Forward normal pour les autres couches
                    x_residual = self.transformer._start_residual(x, layer, 0)
                    x_residual = layer['attention'](x_residual, x_residual, None, None)
                    x = self.transformer._end_residual(x, x_residual, layer, 0)
                    
                    x_residual = self.transformer._start_residual(x, layer, 1)
                    x_residual = layer['linear0'](x_residual)
                    x_residual = self.transformer.activation(x_residual)
                    if self.transformer.ffn_dropout:
                        x_residual = F.dropout(x_residual, self.transformer.ffn_dropout, self.training)
                    x_residual = layer['linear1'](x_residual)
                    x = self.transformer._end_residual(x, x_residual, layer, 1)
            
            # Mapper les scores aux noms de features
            importance = {}
            
            # CLS token (index 0) - ignoré
            token_idx = 1
            
            # Features numériques
            for i, name in enumerate(feature_names):
                if token_idx < len(cls_attention):
                    importance[name] = cls_attention[token_idx].item()
                    token_idx += 1
            
            # Features catégorielles (si présentes)
            if x_cat is not None and x_cat.shape[1] > 0:
                for i in range(x_cat.shape[1]):
                    cat_name = f"cat_feature_{i}"
                    if token_idx < len(cls_attention):
                        importance[cat_name] = cls_attention[token_idx].item()
                        token_idx += 1
            
            return importance


def load_california_housing():
    """Charge et prépare le dataset California Housing."""
    data = fetch_california_housing()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    # Standardiser les features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convertir en tenseurs PyTorch
    X_num = torch.FloatTensor(X_scaled)
    X_cat = torch.zeros(X_num.shape[0], 0, dtype=torch.long)
    y = torch.FloatTensor(y)
    
    return X_num, X_cat, y, list(feature_names)


def split_data(X_num: torch.Tensor, X_cat: torch.Tensor, y: torch.Tensor, 
               train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict[str, Tuple[torch.Tensor, ...]]:
    """Divise les données en ensembles d'entraînement, validation et test."""
    n_samples = X_num.shape[0]
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    return {
        'train': (X_num[train_indices], X_cat[train_indices], y[train_indices]),
        'val': (X_num[val_indices], X_cat[val_indices], y[val_indices]),
        'test': (X_num[test_indices], X_cat[test_indices], y[test_indices])
    }


def create_data_loaders(data_splits: Dict, batch_size: int = 256):
    """Crée les DataLoaders pour l'entraînement."""
    from torch.utils.data import TensorDataset, DataLoader
    
    loaders = {}
    for split_name, (X_num, X_cat, y) in data_splits.items():
        dataset = TensorDataset(X_num, X_cat, y)
        shuffle = (split_name == 'train')
        loaders[split_name] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loaders


def train_epoch(model, optimizer, data_loader, loss_fn, device):
    """Entraîne le modèle pour une époque."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for X_num_batch, X_cat_batch, y_batch in data_loader:
        X_num_batch = X_num_batch.to(device)
        X_cat_batch = X_cat_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        predictions = model(X_num_batch, X_cat_batch)
        loss = loss_fn(predictions, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate_model(model, data_loader, loss_fn, device):
    """Évalue le modèle sur un ensemble de données."""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    num_batches = 0
    
    with torch.no_grad():
        for X_num_batch, X_cat_batch, y_batch in data_loader:
            X_num_batch = X_num_batch.to(device)
            X_cat_batch = X_cat_batch.to(device)
            y_batch = y_batch.to(device)
            
            pred_batch = model(X_num_batch, X_cat_batch)
            loss = loss_fn(pred_batch, y_batch)
            
            total_loss += loss.item()
            predictions.append(pred_batch.cpu())
            targets.append(y_batch.cpu())
            num_batches += 1
    
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    
    return total_loss / num_batches, predictions, targets


def main():
    """Fonction principale d'entraînement."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== FT-Transformer PLR - California Housing ===")
    print(f"Device: {device}")
    
    # Configuration
    config = {
        'batch_size': 256,
        'n_epochs': 50,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'patience': 15,
        'loss_type': 'mse',
        'normalize_targets': True,
        'seed': 42
    }
    
    # Paramètres du modèle FT-Transformer PLR
    model_config = {
        'd_token': 64,
        'n_blocks': 3,
        'n_heads': 8,
        'attention_dropout': 0.1,
        'ffn_d_hidden': 256,
        'ffn_dropout': 0.1,
        'residual_dropout': 0.0,
        'd_out': 1,
        'average_layer': True
    }
    
    print(f"Configuration: {config}")
    print(f"Modèle: {model_config}")
    
    # Fixer la seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Charger les données
    print("\n=== Chargement des données ===")
    X_num, X_cat, y, feature_names = load_california_housing()
    print(f"Données: {X_num.shape[0]} échantillons, {X_num.shape[1]} features")
    
    # Diviser les données
    data_splits = split_data(X_num, X_cat, y)
    print(f"Train: {data_splits['train'][0].shape[0]}, Val: {data_splits['val'][0].shape[0]}, Test: {data_splits['test'][0].shape[0]}")
    
    # Normaliser les cibles
    if config['normalize_targets']:
        y_train_norm, y_mean, y_std = normalize_targets(data_splits['train'][2])
        y_val_norm = (data_splits['val'][2] - y_mean) / y_std
        y_test_norm = (data_splits['test'][2] - y_mean) / y_std
        
        data_splits['train'] = (data_splits['train'][0], data_splits['train'][1], y_train_norm)
        data_splits['val'] = (data_splits['val'][0], data_splits['val'][1], y_val_norm)
        data_splits['test'] = (data_splits['test'][0], data_splits['test'][1], y_test_norm)
        
        print(f"Cibles normalisées (mean={y_mean.item():.3f}, std={y_std.item():.3f})")
    
    # Créer les DataLoaders
    data_loaders = create_data_loaders(data_splits, config['batch_size'])
    
    # Créer le modèle avec les embeddings P-LR
    print("\n=== Création du modèle FT-Transformer PLR ===")
    model = InterpretableFTTransformerPLRRegression.make_baseline(
        n_num_features=X_num.shape[1],
        cat_cardinalities=[],
        X_train=data_splits['train'][0],  # Passer les données d'entraînement pour P-LR
        **model_config
    )
    model.to(device)
    
    print(f"Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ Embeddings P-LR initialisés avec les données d'entraînement")
    
    # Optimiseur et fonction de perte
    optimizer = torch.optim.AdamW(model.optimization_param_groups(), 
                                  lr=config['lr'], weight_decay=config['weight_decay'])
    loss_fn = get_regression_loss_function(config['loss_type'])
    
    # Variables pour le suivi
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    
    print(f"\n=== Entraînement ({config['n_epochs']} époques max) ===")
    
    # Boucle d'entraînement
    for epoch in range(config['n_epochs']):
        start_time = time.time()
        
        train_loss = train_epoch(model, optimizer, data_loaders['train'], loss_fn, device)
        val_loss, val_predictions, val_targets = evaluate_model(model, data_loaders['val'], loss_fn, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {epoch_time:.2f}s')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f' <<< NOUVEAU MEILLEUR MODÈLE')
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f'\nEarly stopping à l\'époque {epoch}')
                break
    
    # Charger le meilleur modèle
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Évaluation finale
    print(f"\n=== Évaluation finale ===")
    
    val_loss, val_predictions, val_targets = evaluate_model(model, data_loaders['val'], loss_fn, device)
    test_loss, test_predictions, test_targets = evaluate_model(model, data_loaders['test'], loss_fn, device)
    
    # Dénormaliser
    if config['normalize_targets']:
        val_predictions = denormalize_predictions(val_predictions, y_mean, y_std)
        val_targets = denormalize_predictions(val_targets, y_mean, y_std)
        test_predictions = denormalize_predictions(test_predictions, y_mean, y_std)
        test_targets = denormalize_predictions(test_targets, y_mean, y_std)
    
    # Calculer les métriques
    val_metrics = regression_performance_dict(val_targets.numpy(), val_predictions.numpy())
    test_metrics = regression_performance_dict(test_targets.numpy(), test_predictions.numpy())
    
    print("\n--- Résultats Validation ---")
    print_regression_metrics(val_metrics)
    
    print("\n--- Résultats Test ---")
    print_regression_metrics(test_metrics)
    
    # Analyse d'interprétabilité
    print(f"\n=== Analyse d'interprétabilité ===")
    
    X_dict = {'test': (data_splits['test'][0], data_splits['test'][1])}
    y_dict = {'test': data_splits['test'][2]}
    
    performance_results = {
        'val': list(val_metrics.values()),
        'test': list(test_metrics.values())
    }
    
    training_results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'total_epochs': len(train_losses)
    }
    
    interpretability_results = analyze_interpretability(
        model=model,
        X=X_dict,
        y=y_dict,
        model_name="ft_transformer_plr_regression",
        seed=config['seed'],
        model_config=model_config,
        training_results=training_results,
        performance_results=performance_results,
        feature_names=feature_names,
        results_base_dir="results"
    )
    
    print("=== FT-Transformer PLR terminé ===")


if __name__ == '__main__':
    main()
    main()
