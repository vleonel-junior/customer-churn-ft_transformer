import rtdl_lib
from rtdl_lib.modules import FTTransformer
import zero
from data.process_telecom_data import device, get_data
from train_funct import train, val, evaluate
import numpy as np
import time
import torch
import os
from itertools import product
from num_embedding_factory import get_num_embedding
import json

# Paramètres de grid search
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
    "embedding_type": [
        "L", "LR", "LR-LR",
        "Q", "Q-L", "Q-LR", "Q-LR-LR",
        "T", "T-L", "T-LR", "T-LR-LR",
        "P", "P-L", "P-LR", "P-LR-LR"
    ],
}
seeds = [0, 1, 2, 3, 4]
d_out = 1
patience = 20

metrics_dir = "results/results_telecom/métriques/"
os.makedirs(metrics_dir, exist_ok=True)

keys, values = zip(*param_grid.items())
grid = [dict(zip(keys, v)) for v in product(*values)]

def to_named_dict(values):
    metric_names = [
        "roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "mcc",
        "sensitivity", "specificity", "precision", "f1", "cohen_kappa"
    ]
    if isinstance(values, dict):
        return values
    return {name: float(val) for name, val in zip(metric_names, values)}

for grid_idx, grid_params in enumerate(grid):
    all_results = []
    for seed in seeds:
        # Préparation des dossiers de sortie
        output_dir = f'results/results_telecom/ftt_gridsearch/seed_{seed}/grid_{grid_idx}'
        os.makedirs(f"{output_dir}/heatmaps", exist_ok=True)
        os.makedirs(f"{output_dir}/best_models", exist_ok=True)
        os.makedirs(f"{output_dir}/métriques", exist_ok=True)

        # Chargement des données
        X, y, cat_cardinalities = get_data(seed)
        batch_size = grid_params["batch_size"]
        n_epochs = grid_params["n_epochs"]

        train_loader = zero.data.IndexLoader(len(y['train']), batch_size, device=device)
        val_loader = zero.data.IndexLoader(len(y['val']), batch_size, device=device)

        # Modèle FTTransformer
        model = FTTransformer.make_default(
            n_num_features=X['train'][0].shape[1],
            cat_cardinalities=cat_cardinalities,
            last_layer_query_idx=[-1],
            d_out=d_out,
            n_blocks=grid_params["n_blocks"],
            d_token=grid_params["d_token"],
            attention_dropout=grid_params["attention_dropout"],
            ffn_d_hidden=grid_params["ffn_d_hidden"],
            ffn_dropout=grid_params["ffn_dropout"],
            residual_dropout=grid_params["residual_dropout"],
            attention_n_heads=grid_params["attention_n_heads"],
        )

        d_embedding = model.feature_tokenizer.d_token
        embedding_type = grid_params["embedding_type"]

        # Embedding numérique personnalisé
        X_train_cpu = X['train'][0].cpu()
        num_embedding = get_num_embedding(
            embedding_type=embedding_type,
            X_train=X_train_cpu,
            d_embedding=d_embedding,
            y_train=y['train'] if embedding_type.startswith("T") else None
        )
        model.feature_tokenizer.num_tokenizer = num_embedding

        model.to(device)

        optimizer = (
            model.make_default_optimizer()
            if hasattr(model, "make_default_optimizer")
            else torch.optim.AdamW(model.parameters(), lr=grid_params["lr"], weight_decay=grid_params["weight_decay"])
        )

        loss_fn = torch.nn.BCEWithLogitsLoss()

        train_loss_list = []
        val_loss_list = []
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_model_state = None
        best_metrics = None

        for epoch in range(n_epochs):
            start_time = time.time()
            train_loss = train(epoch, model, optimizer, X, y, train_loader, loss_fn)
            val_loss = val(epoch, model, X, y, val_loader, loss_fn)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            epoch_time = time.time() - start_time

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                test_performance = evaluate(model, 'test', X, y, seed)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Charger le meilleur modèle
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        val_performance = evaluate(model, 'val', X, y, seed)
        test_performance = evaluate(model, 'test', X, y, seed)

        result = {
            "seed": seed,
            **grid_params,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "test_performance": to_named_dict(test_performance),
            "val_performance": to_named_dict(val_performance),
            "train_losses": train_loss_list,
            "val_losses": val_loss_list,
        }
        all_results.append(result)

        # Sauvegarde des résultats pour ce seed et cette config
        with open(f'{output_dir}/métriques/ftt_training_results.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        torch.save(model.state_dict(), f'{output_dir}/best_models/ftt_best_model.pt')

    # Sauvegarde JSON pour ce grid complet
    json_path = os.path.join(metrics_dir, f"ftt_grid{grid_idx}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Résultats sauvegardés dans {json_path}")