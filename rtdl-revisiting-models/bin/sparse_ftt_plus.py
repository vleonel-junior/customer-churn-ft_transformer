# %%
import math
import typing as ty
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
import zero
from torch import Tensor

# Ajouter le chemin vers le dossier parent pour importer nos modèles
sys.path.append(str(Path(__file__).parent.parent.parent))

import lib
from sparse_ftt_plus.attention import InterpretableMultiHeadAttention
from rtdl_lib.modules import FeatureTokenizer, CLSToken, _make_nn_module

ModuleType = ty.Union[str, ty.Callable[..., nn.Module]]


# %%
class SparseFTTPlus(nn.Module):
    """Sparse FT-Transformer Plus avec attention interprétable et sparsemax.
    
    Cette implémentation combine l'architecture FT-Transformer de RTDL avec les
    innovations FTT+ d'Isomura et al. pour une attention sélective et interprétable.
    
    Le modèle gère automatiquement la classification et la régression selon le dataset.
    
    References:
    - Gorishniy et al. "Revisiting Deep Learning Models for Tabular Data" (NeurIPS 2021)
    - Isomura et al. "Optimizing FT-Transformer: Sparse Attention for Improved Performance and Interpretability" (2023)
    """

    def __init__(
        self,
        *,
        # Architecture de base
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        n_blocks: int,
        n_heads: int,
        # Attention configuration
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: ModuleType,
        attention_mode: str,
        # FFN configuration
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: ModuleType,
        ffn_normalization: ModuleType,
        # Autres paramètres
        residual_dropout: float,
        prenormalization: bool,
        head_activation: ModuleType,
        head_normalization: ModuleType,
        d_out: int,
    ) -> None:
        super().__init__()
        
        # Feature Tokenizer RTDL pour l'embedding optimal
        self.feature_tokenizer = FeatureTokenizer(
            n_num_features=d_numerical,
            cat_cardinalities=categories or [],
            d_token=d_token
        )
        
        # Token CLS pour l'inférence BERT-like
        self.cls_token = CLSToken(d_token, self.feature_tokenizer.initialization)
        
        # Blocs Transformer avec attention interprétable
        self.blocks = nn.ModuleList([
            self._make_transformer_block(
                d_token=d_token,
                n_heads=n_heads,
                attention_dropout=attention_dropout,
                attention_initialization=attention_initialization,
                attention_normalization=attention_normalization,
                attention_mode=attention_mode,
                ffn_d_hidden=ffn_d_hidden,
                ffn_dropout=ffn_dropout,
                ffn_activation=ffn_activation,
                ffn_normalization=ffn_normalization,
                residual_dropout=residual_dropout,
                prenormalization=prenormalization,
            )
            for _ in range(n_blocks)
        ])
        
        # Configuration
        self.prenormalization = prenormalization
        self.residual_dropout = residual_dropout
        
        # Normalisation finale et tête de prédiction
        from rtdl_lib.modules import Transformer
        self.last_normalization = (
            _make_nn_module(head_normalization, d_token) 
            if prenormalization else None
        )
        self.head_activation = lib.get_nonglu_activation_fn(head_activation)
        self.head = nn.Linear(d_token, d_out)
    
    def _make_transformer_block(
        self,
        d_token: int,
        n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: ModuleType,
        attention_mode: str,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: ModuleType,
        ffn_normalization: ModuleType,
        residual_dropout: float,
        prenormalization: bool,
    ) -> nn.ModuleDict:
        """Crée un bloc Transformer avec attention interprétable."""
        
        block = nn.ModuleDict({
            'attention': InterpretableMultiHeadAttention(
                d_model=d_token,
                n_heads=n_heads,
                dropout=attention_dropout,
                initialization=attention_initialization,
                attention_mode=attention_mode
            ),
            'norm1': _make_nn_module(attention_normalization, d_token),
        })
        
        if not prenormalization:
            block['norm0'] = _make_nn_module(attention_normalization, d_token)
        
        # FFN utilisant l'implémentation RTDL
        from rtdl_lib.modules import Transformer
        block['ffn'] = Transformer.FFN(
            d_token=d_token,
            d_hidden=ffn_d_hidden,
            bias_first=True,
            bias_second=True,
            dropout=ffn_dropout,
            activation=ffn_activation
        )
        
        block['norm_ffn'] = _make_nn_module(ffn_normalization, d_token)
        
        return block
    
    def _start_residual(self, x: Tensor, block: nn.ModuleDict, stage: str) -> Tensor:
        """Démarre une connexion résiduelle avec normalisation pré/post."""
        if self.prenormalization:
            if stage == 'attention':
                return block.get('norm0', lambda x: x)(x)
            else:  # stage == 'ffn'
                return block['norm1'](x)
        return x
    
    def _end_residual(self, x: Tensor, x_residual: Tensor, block: nn.ModuleDict, stage: str) -> Tensor:
        """Termine une connexion résiduelle avec dropout et normalisation."""
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        
        x = x + x_residual
        
        if not self.prenormalization:
            if stage == 'attention':
                x = block.get('norm0', lambda x: x)(x)
            else:  # stage == 'ffn'
                x = block['norm1'](x)
        
        return x

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        """Forward pass unifié pour classification et régression."""
        # Tokenisation des features
        x = self.feature_tokenizer(x_num, x_cat)
        
        # Ajout du token CLS
        x = self.cls_token(x)
        
        # Passage à travers les blocs Transformer
        for block_idx, block in enumerate(self.blocks):
            is_last_block = block_idx + 1 == len(self.blocks)
            
            # Bloc d'attention avec connexion résiduelle
            x_residual = self._start_residual(x, block, 'attention')
            
            # Pour le dernier bloc, on peut optimiser en ne traitant que le token CLS
            if is_last_block:
                x_residual, _ = block['attention'](x_residual[:, :1], x_residual)
                x = x[:, :x_residual.shape[1]]  # Ajuster la taille
            else:
                x_residual, _ = block['attention'](x_residual)
            
            x = self._end_residual(x, x_residual, block, 'attention')
            
            # Bloc FFN avec connexion résiduelle
            x_residual = self._start_residual(x, block, 'ffn')
            x_residual = block['ffn'](x_residual)
            x = self._end_residual(x, x_residual, block, 'ffn')
        
        # Extraction du token CLS et prédiction finale
        assert x.shape[1] == 1, f"Expected 1 token, got {x.shape[1]}"
        x = x[:, 0]  # Extraire le token CLS
        
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        
        x = self.head_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)
        
        return x


# %%
if __name__ == "__main__":
    args, output = lib.load_config()
    
    # Paramètres par défaut pour Sparse FTT+
    args['model'].setdefault('attention_mode', 'hybrid')
    args['model'].setdefault('attention_initialization', 'kaiming')
    args['model'].setdefault('attention_normalization', 'LayerNorm')
    args['model'].setdefault('ffn_activation', 'ReGLU')
    args['model'].setdefault('ffn_normalization', 'LayerNorm')
    args['model'].setdefault('prenormalization', True)
    args['model'].setdefault('head_activation', 'ReLU')
    args['model'].setdefault('head_normalization', 'LayerNorm')

    # %%
    zero.set_randomness(args['seed'])
    dataset_dir = lib.get_path(args['data']['path'])
    stats: ty.Dict[str, ty.Any] = {
        'dataset': dataset_dir.name,
        'algorithm': Path(__file__).stem,
        **lib.load_json(output / 'stats.json'),
    }
    timer = zero.Timer()
    timer.run()

    D = lib.Dataset.from_dir(dataset_dir)
    X = D.build_X(
        normalization=args['data'].get('normalization'),
        num_nan_policy='mean',
        cat_nan_policy='new',
        cat_policy=args['data'].get('cat_policy', 'indices'),
        cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
        seed=args['seed'],
    )
    if not isinstance(X, tuple):
        X = (X, None)
    zero.set_randomness(args['seed'])
    Y, y_info = D.build_y(args['data'].get('y_policy'))
    lib.dump_pickle(y_info, output / 'y_info.pickle')
    X = tuple(None if x is None else lib.to_tensors(x) for x in X)
    Y = lib.to_tensors(Y)
    device = lib.get_device()
    if device.type != 'cpu':
        X = tuple(
            None if x is None else {k: v.to(device) for k, v in x.items()} for x in X
        )
        Y_device = {k: v.to(device) for k, v in Y.items()}
    else:
        Y_device = Y
    X_num, X_cat = X
    del X
    if not D.is_multiclass:
        Y_device = {k: v.float() for k, v in Y_device.items()}

    train_size = D.size(lib.TRAIN)
    batch_size = args['training']['batch_size']
    epoch_size = stats['epoch_size'] = math.ceil(train_size / batch_size)
    eval_batch_size = args['training']['eval_batch_size']
    chunk_size = None

    loss_fn = (
        F.binary_cross_entropy_with_logits
        if D.is_binclass
        else F.cross_entropy
        if D.is_multiclass
        else F.mse_loss
    )
    
    # Créer le modèle Sparse FTT+
    model = SparseFTTPlus(
        d_numerical=0 if X_num is None else X_num['train'].shape[1],
        categories=lib.get_categories(X_cat),
        d_out=D.info['n_classes'] if D.is_multiclass else 1,
        **args['model'],
    ).to(device)
    
    if torch.cuda.device_count() > 1:  # type: ignore[code]
        print('Using nn.DataParallel')
        model = nn.DataParallel(model)
    stats['n_parameters'] = lib.get_n_parameters(model)

    # Groupes de paramètres optimisés (style RTDL)
    def needs_wd(name):
        return all(x not in name for x in ['feature_tokenizer', '.norm', '.bias'])

    for x in ['feature_tokenizer', '.norm', '.bias']:
        assert any(x in a for a in (b[0] for b in model.named_parameters()))
    parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
    parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
    optimizer = lib.make_optimizer(
        args['training']['optimizer'],
        (
            [
                {'params': parameters_with_wd},
                {'params': parameters_without_wd, 'weight_decay': 0.0},
            ]
        ),
        args['training']['lr'],
        args['training']['weight_decay'],
    )

    stream = zero.Stream(lib.IndexLoader(train_size, batch_size, True, device))
    progress = zero.ProgressTracker(args['training']['patience'])
    training_log = {lib.TRAIN: [], lib.VAL: [], lib.TEST: []}
    timer = zero.Timer()
    checkpoint_path = output / 'checkpoint.pt'

    def print_epoch_info():
        print(f'\n>>> Epoch {stream.epoch} | {lib.format_seconds(timer())} | {output}')
        print(
            ' | '.join(
                f'{k} = {v}'
                for k, v in {
                    'lr': lib.get_lr(optimizer),
                    'batch_size': batch_size,
                    'chunk_size': chunk_size,
                    'epoch_size': stats['epoch_size'],
                    'n_parameters': stats['n_parameters'],
                }.items()
            )
        )

    def apply_model(part, idx):
        return model(
            None if X_num is None else X_num[part][idx],
            None if X_cat is None else X_cat[part][idx],
        )

    @torch.no_grad()
    def evaluate(parts):
        global eval_batch_size
        model.eval()
        metrics = {}
        predictions = {}
        for part in parts:
            while eval_batch_size:
                try:
                    predictions[part] = (
                        torch.cat(
                            [
                                apply_model(part, idx)
                                for idx in lib.IndexLoader(
                                    D.size(part), eval_batch_size, False, device
                                )
                            ]
                        )
                        .cpu()
                        .numpy()
                    )
                except RuntimeError as err:
                    if not lib.is_oom_exception(err):
                        raise
                    eval_batch_size //= 2
                    print('New eval batch size:', eval_batch_size)
                    stats['eval_batch_size'] = eval_batch_size
                else:
                    break
            if not eval_batch_size:
                RuntimeError('Not enough memory even for eval_batch_size=1')
            metrics[part] = lib.calculate_metrics(
                D.info['task_type'],
                Y[part].numpy(),  # type: ignore[code]
                predictions[part],  # type: ignore[code]
                'logits',
                y_info,
            )
        for part, part_metrics in metrics.items():
            print(f'[{part:<5}]', lib.make_summary(part_metrics))
        return metrics, predictions

    def save_checkpoint(final):
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'stream': stream.state_dict(),
                'random_state': zero.get_random_state(),
                **{
                    x: globals()[x]
                    for x in [
                        'progress',
                        'stats',
                        'timer',
                        'training_log',
                    ]
                },
            },
            checkpoint_path,
        )
        lib.dump_stats(stats, output, final)
        lib.backup_output(output)

    # %%
    timer.run()
    for epoch in stream.epochs(args['training']['n_epochs']):
        print_epoch_info()

        model.train()
        epoch_losses = []
        for batch_idx in epoch:
            loss, new_chunk_size = lib.train_with_auto_virtual_batch(
                optimizer,
                loss_fn,
                lambda x: (apply_model(lib.TRAIN, x), Y_device[lib.TRAIN][x]),
                batch_idx,
                chunk_size or batch_size,
            )
            epoch_losses.append(loss.detach())
            if new_chunk_size and new_chunk_size < (chunk_size or batch_size):
                stats['chunk_size'] = chunk_size = new_chunk_size
                print('New chunk size:', chunk_size)
        epoch_losses = torch.stack(epoch_losses).tolist()
        training_log[lib.TRAIN].extend(epoch_losses)
        print(f'[{lib.TRAIN}] loss = {round(sum(epoch_losses) / len(epoch_losses), 3)}')

        metrics, predictions = evaluate([lib.VAL, lib.TEST])
        for k, v in metrics.items():
            training_log[k].append(v)
        progress.update(metrics[lib.VAL]['score'])

        if progress.success:
            print('New best epoch!')
            stats['best_epoch'] = stream.epoch
            stats['metrics'] = metrics
            save_checkpoint(False)
            for k, v in predictions.items():
                np.save(output / f'p_{k}.npy', v)

        elif progress.fail:
            break

    # %%
    print('\nRunning the final evaluation...')
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    stats['metrics'], predictions = evaluate(lib.PARTS)
    for k, v in predictions.items():
        np.save(output / f'p_{k}.npy', v)
    stats['time'] = lib.format_seconds(timer())
    save_checkpoint(True)
    print('Done!')