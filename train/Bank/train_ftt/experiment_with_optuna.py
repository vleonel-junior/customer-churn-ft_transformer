import os
import json
import numpy as np
import torch
import optuna
from train_funct import train, val, evaluate
from data.process_bank_data import device, get_data
import zero
import rtdl_lib
from rtdl_lib.modules import FTTransformer
from num_embedding_factory import get_num_embedding
import gc

# --- Paramètres fixes ---
seeds = [0, 1, 2]
metrics_dir = "results/results_bank/ftt_optuna/"
os.makedirs(metrics_dir, exist_ok=True)

def to_named_dict(values):
    metric_names = [
        "roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "mcc",
        "sensitivity", "specificity", "precision", "f1", "cohen_kappa"
    ]
    if isinstance(values, dict):
        return values
    return {name: float(val) for name, val in zip(metric_names, values)}

def objective(trial):
    """Fonction objectif optimisée pour Optuna"""
    # 1. Hyperparamètres avec espaces de recherche étendus
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    num_embedding_type = trial.suggest_categorical(
        "num_embedding_type",
        [
            "L", "P-LR", "P-LR-LR"
        ]
    )
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8, 16])
    d_embedding = trial.suggest_categorical("d_embedding", [16, 32, 64, 128])
    n_layers = trial.suggest_int("n_layers", 2, 6)
    attention_dropout = trial.suggest_float("attention_dropout", 0.1, 0.3)
    ffn_dropout = trial.suggest_float("ffn_dropout", 0.1, 0.3)
    residual_dropout = trial.suggest_float("residual_dropout", 0.1, 0.2)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    patience_epochs = trial.suggest_int("patience", 15, 30)
    min_delta = 1e-4

    try:
        aucs = []
        pr_aucs = []
        all_seed_results = []

        for seed_idx, seed in enumerate(seeds):
            print(f"Trial {trial.number}, Seed {seed_idx+1}/{len(seeds)}")

            X, y, cat_cardinalities = get_data(seed)

            train_loader = zero.data.IndexLoader(len(y['train']), batch_size, device=device)
            val_loader = zero.data.IndexLoader(len(y['val']), batch_size, device=device)

            # Embedding numérique sur le bon device
            X_train_device = X['train'][0].to(device)
            num_embedding = get_num_embedding(
                embedding_type=num_embedding_type,
                X_train=X_train_device,
                d_embedding=d_embedding,
                y_train=y['train'].to(device) if num_embedding_type.startswith("T") else None
            )

            # Modèle FTTransformer sur le bon device
            model = FTTransformer.make_baseline(
                n_num_features=X['train'][0].shape[1],
                cat_cardinalities=cat_cardinalities,
                d_token=d_embedding,
                n_blocks=n_layers,
                attention_dropout=attention_dropout,
                ffn_d_hidden=d_embedding * 2,
                ffn_dropout=ffn_dropout,
                residual_dropout=residual_dropout,
                last_layer_query_idx=[-1],
                d_out=1,
            )
            model.feature_tokenizer.num_tokenizer = num_embedding
            model.to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5, verbose=False
            )
            loss_fn = torch.nn.BCEWithLogitsLoss()

            best_val_loss = float('inf')
            best_metrics = None
            patience_counter = 0
            best_model_state = None
            train_loss_list = []
            val_loss_list = []
            best_epoch = 0

            reported_steps = set()
            for epoch in range(100):
                loss_train = train(epoch, model, optimizer, X, y, train_loader, loss_fn)
                loss_val = val(epoch, model, X, y, val_loader, loss_fn)
                train_loss_list.append(loss_train)
                val_loss_list.append(loss_val)

                scheduler.step(loss_val)

                if epoch not in reported_steps:
                    trial.report(loss_val, epoch)
                    reported_steps.add(epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                if loss_val < best_val_loss - min_delta:
                    best_val_loss = loss_val
                    patience_counter = 0
                    best_epoch = epoch
                    best_model_state = model.state_dict().copy()
                    metrics = evaluate(model, 'test', X, y, seed)
                    best_metrics = metrics
                else:
                    patience_counter += 1
                    if patience_counter >= patience_epochs:
                        print(f"Early stopping at epoch {epoch}")
                        break

            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            val_performance = evaluate(model, 'val', X, y, seed)
            test_performance = evaluate(model, 'test', X, y, seed)

            aucs.append(best_metrics[0] if best_metrics is not None else 0)
            pr_aucs.append(best_metrics[1] if best_metrics is not None else 0)

            # Sauvegarde détaillée par seed/trial
            output_dir = os.path.join(metrics_dir, f"trial_{trial.number}", f"seed_{seed}")
            os.makedirs(f"{output_dir}/best_models", exist_ok=True)
            os.makedirs(f"{output_dir}/métriques", exist_ok=True)
            result = {
                "trial_number": trial.number,
                "seed": seed,
                "lr": lr,
                "weight_decay": weight_decay,
                "num_embedding_type": num_embedding_type,
                "n_heads": n_heads,
                "d_embedding": d_embedding,
                "n_layers": n_layers,
                "attention_dropout": attention_dropout,
                "ffn_dropout": ffn_dropout,
                "residual_dropout": residual_dropout,
                "batch_size": batch_size,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "test_performance": to_named_dict(test_performance),
                "val_performance": to_named_dict(val_performance),
                "train_losses": train_loss_list,
                "val_losses": val_loss_list,
            }
            all_seed_results.append(result)
            with open(f'{output_dir}/métriques/ftt_training_results.json', 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            torch.save(model.state_dict(), f'{output_dir}/best_models/ftt_best_model.pt')

            # Nettoyage mémoire explicite
            torch.cuda.empty_cache()
            gc.collect()

        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        mean_pr_auc = np.mean(pr_aucs)

        trial.set_user_attr("detailed_results", {
            "hyperparams": {
                "lr": lr,
                "weight_decay": weight_decay,
                "num_embedding_type": num_embedding_type,
                "n_heads": n_heads,
                "d_embedding": d_embedding,
                "n_layers": n_layers,
                "attention_dropout": attention_dropout,
                "ffn_dropout": ffn_dropout,
                "residual_dropout": residual_dropout,
                "batch_size": batch_size,
            },
            "results": {
                "aucs_per_seed": aucs,
                "pr_aucs_per_seed": pr_aucs,
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "mean_pr_auc": mean_pr_auc,
                "seeds": seeds,
                "all_seed_results": all_seed_results
            }
        })

        # Sauvegarde intermédiaire globale
        with open(os.path.join(metrics_dir, f"trial_{trial.number}_all_seeds.json"), "w") as f:
            json.dump(all_seed_results, f, indent=2, ensure_ascii=False)

        return mean_auc

    except Exception as e:
        print(f"Error in trial {trial.number}: {str(e)}")
        raise

if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=10,
        n_ei_candidates=24,
        seed=42
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
        interval_steps=5
    )

    study = optuna.create_study(
        direction="maximize",
        study_name="ftt_optuna_enhanced",
        sampler=sampler,
        pruner=pruner
    )

    def save_callback(study, trial):
        if trial.number % 5 == 0:
            with open(os.path.join(metrics_dir, "intermediate_results.json"), "w") as f:
                results = []
                for t in study.trials:
                    if t.value is not None:
                        result = {"trial_number": t.number, "value": t.value}
                        result.update(t.params)
                        if "detailed_results" in t.user_attrs:
                            result.update(t.user_attrs["detailed_results"])
                        results.append(result)
                json.dump(results, f, indent=2)

    try:
        study.optimize(
            objective,
            n_trials=50,
            callbacks=[save_callback],
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("Optimization interrupted by user")

    best_trial = study.best_trial

    with open(os.path.join(metrics_dir, "best_params.json"), "w") as f:
        json.dump(best_trial.params, f, indent=2)

    if "detailed_results" in best_trial.user_attrs:
        with open(os.path.join(metrics_dir, "best_detailed_results.json"), "w") as f:
            json.dump(best_trial.user_attrs["detailed_results"], f, indent=2)

    all_trials = []
    for t in study.trials:
        if t.value is not None:
            trial_data = {
                "trial_number": t.number,
                "value": t.value,
                "params": t.params,
                "state": t.state.name
            }
            if "detailed_results" in t.user_attrs:
                trial_data["detailed_results"] = t.user_attrs["detailed_results"]
            all_trials.append(trial_data)

    with open(os.path.join(metrics_dir, "all_trials_detailed.json"), "w") as f:
        json.dump(all_trials, f, indent=2)

    print(f"Optimization completed!")
    print(f"Best trial: {best_trial.number}")
    print(f"Best mean AUC: {best_trial.value:.4f}")
    print(f"Best params: {best_trial.params}")

    if len(study.trials) > 10:
        importance = optuna.importance.get_param_importances(study)
        print("Parameter importance:")
        for param, imp in importance.items():
            print(f"  {param}: {imp:.4f}")

        with open(os.path.join(metrics_dir, "param_importance.json"), "w") as f:
            json.dump(importance, f, indent=2)