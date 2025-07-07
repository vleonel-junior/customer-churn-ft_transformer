"""
Script d'expérimentation pour FTT (multi-seeds, grid search, logs détaillés)
Résultats enregistrés dans results/results_telecom/métriques/
Format de sauvegarde : JSON
"""

import os
import json
import numpy as np
import torch
import zero
from train_funct import train, val, evaluate
from data.process_telecom_data import device, get_data
import sys
sys.path.append('../../../rtdl_lib')
import rtdl

# Paramètres de base
d_out = 1
lr = 0.001
weight_decay = 0.0
batch_size = 64
n_epochs = 50
seeds = [0, 1, 2, 3, 4]
metrics_dir = "results/results_telecom/métriques/"
os.makedirs(metrics_dir, exist_ok=True)

# (Optionnel) Grid search sur d'autres hyperparamètres
from itertools import product

param_grid = {
    "lr": [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.008, 0.01, 0.02, 0.05],
    "weight_decay": [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    "batch_size": [16, 32, 48, 64, 96, 128, 192, 256],
    "n_epochs": [30, 50, 75, 100, 150, 200],
    "d_token": [32, 64, 96, 128, 192, 256, 384],
    "n_blocks": [1, 2, 3, 4, 5, 6, 8],
    "attention_n_heads": [2, 4, 6, 8, 12, 16],
    "attention_dropout": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
    "ffn_d_hidden": [64, 128, 256, 384, 512, 768, 1024],
    "ffn_dropout": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
    "residual_dropout": [0.0, 0.05, 0.1, 0.15, 0.2],
    "embedding_type": ["linear", "mlp", "sinusoidal", "learnable"],
}

keys, values = zip(*param_grid.items())
grid = [dict(zip(keys, v)) for v in product(*values)]

for grid_idx, grid_params in enumerate(grid):
    all_results = []
    for seed in seeds:
        # Chargement des données
        X, y, cat_cardinalities = get_data(seed)
        train_loader = zero.data.IndexLoader(len(y['train']), batch_size, device=device)
        val_loader = zero.data.IndexLoader(len(y['val']), batch_size, device=device)
        test_loader = zero.data.IndexLoader(len(y['test']), batch_size, device=device)

        # Modèle
        model = rtdl.FTTransformer.make_default(
            n_num_features=X['train'][0].shape[1],
            cat_cardinalities=cat_cardinalities,
            last_layer_query_idx=[-1],
            d_out=d_out,
        )
        model.to(device)
        optimizer = (
            model.make_default_optimizer()
            if isinstance(model, rtdl.FTTransformer)
            else torch.optim.AdamW(model.parameters(), lr=grid_params["lr"], weight_decay=grid_params["weight_decay"])
        )
        loss_fn = torch.nn.BCELoss()

        # Entraînement
        train_loss_list = []
        val_loss_list = []
        val_loss_check = 10
        best_metrics = None
        for epoch in range(n_epochs):
            loss_train = train(epoch, model, optimizer, X, y, train_loader, loss_fn)
            loss_val = val(epoch, model, X, y, val_loader, loss_fn)
            train_loss_list.append(loss_train)
            val_loss_list.append(loss_val)
            if loss_val <= val_loss_check:
                matrix = evaluate(model, 'test', X, y, seed)
                val_loss_check = loss_val
                best_metrics = matrix

        # Enregistrement des métriques pour ce seed
        # Enregistrer tous les hyperparamètres du grid pour cette config
        result = {
            "seed": seed,
            **grid_params,
            "roc_auc": best_metrics[0],
            "pr_auc": best_metrics[1],
            "acc": best_metrics[2],
            "ba": best_metrics[3],
            "mcc": best_metrics[4],
            "sensitivity": best_metrics[5],
            "specificity": best_metrics[6],
            "precision": best_metrics[7],
            "f1": best_metrics[8],
            "ck": best_metrics[9],
        }
        all_results.append(result)
        print(f"Résultats seed {seed} : {result}")

    # Sauvegarde JSON pour ce grid
    json_path = os.path.join(metrics_dir, f"ftt_grid{grid_idx}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Résultats sauvegardés dans {json_path}")