import os
import json
import numpy as np
import torch
import optuna
import zero
from data.process_telecom_data import device, get_data
from train_func import train, val, evaluate
from ftt_plus.model import InterpretableFTTPlus
from interpretability_analyzer import analyze_interpretability
from num_embedding_factory import get_num_embedding
import gc

# --- Paramètres fixes ---
seeds = [0, 1, 2]
metrics_dir = "results/results_telecom/ftt_plus_optuna/"
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
    # Hyperparamètres à optimiser
    d_token = trial.suggest_categorical("d_token", [16, 32, 64, 128])
    n_blocks = trial.suggest_int("n_blocks", 2, 6)
    ffn_hidden = trial.suggest_categorical("ffn_hidden", [64, 128, 256])
    attention_dropout = trial.suggest_float("attention_dropout", 0.1, 0.3)
    ffn_dropout = trial.suggest_float("ffn_dropout", 0.1, 0.3)
    residual_dropout = trial.suggest_float("residual_dropout", 0.1, 0.2)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    patience = trial.suggest_int("patience", 15, 30)
    embedding_type = trial.suggest_categorical("embedding_type", [
        "L", "LR", "Q", "T", "Q-LR", "T-LR", "P-LR", "P-LR-LR"
    ])

    aucs = []
    pr_aucs = []
    all_seed_results = []

    for seed_idx, seed in enumerate(seeds):
        X, y, cat_cardinalities = get_data(seed)
        train_loader = zero.data.IndexLoader(len(y['train']), batch_size, device=device)
        val_loader = zero.data.IndexLoader(len(y['val']), batch_size, device=device)

        n_num_features = X['train'][0].shape[1]

        # Création du modèle InterpretableFTTPlus
        model = InterpretableFTTPlus.make_baseline(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_token=d_token,
            n_blocks=n_blocks,
            attention_dropout=attention_dropout,
            ffn_d_hidden=ffn_hidden,
            ffn_dropout=ffn_dropout,
            residual_dropout=residual_dropout,
            d_out=1
        )

        # Embedding numérique personnalisé
        X_train_cpu = X['train'][0].cpu()
        num_embedding = get_num_embedding(
            embedding_type=embedding_type,
            X_train=X_train_cpu,
            d_embedding=d_token,
            y_train=y['train'] if embedding_type in ("T", "T-L", "T-LR", "T-LR-LR") else None
        )
        model.feature_tokenizer.num_tokenizer = num_embedding
        model.to(device)

        optimizer = torch.optim.AdamW(model.optimization_param_groups(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        train_loss_list = []
        val_loss_list = []
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_model_state = None
        best_metrics = None

        reported_steps = set()
        for epoch in range(100):
            train_loss = train(epoch, model, optimizer, X, y, train_loader, loss_fn)
            val_loss = val(epoch, model, X, y, val_loader, loss_fn)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

            if epoch not in reported_steps:
                trial.report(val_loss, epoch)
                reported_steps.add(epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                metrics = evaluate(model, 'test', X, y, seed)
                best_metrics = metrics
            else:
                patience_counter += 1
                if patience_counter >= patience:
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
            "d_token": d_token,
            "n_blocks": n_blocks,
            "ffn_hidden": ffn_hidden,
            "attention_dropout": attention_dropout,
            "ffn_dropout": ffn_dropout,
            "residual_dropout": residual_dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "patience": patience,
            "embedding_type": embedding_type,
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

        torch.cuda.empty_cache()
        gc.collect()

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    mean_pr_auc = np.mean(pr_aucs)

    trial.set_user_attr("detailed_results", {
        "hyperparams": {
            "d_token": d_token,
            "n_blocks": n_blocks,
            "ffn_hidden": ffn_hidden,
            "attention_dropout": attention_dropout,
            "ffn_dropout": ffn_dropout,
            "residual_dropout": residual_dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "patience": patience,
            "embedding_type": embedding_type,
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

    with open(os.path.join(metrics_dir, f"trial_{trial.number}_all_seeds.json"), "w") as f:
        json.dump(all_seed_results, f, indent=2, ensure_ascii=False)

    return mean_auc

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
        study_name="ftt_plus_optuna_enhanced",
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
            n_trials=25,
            callbacks=[save_callback],
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        pass

    best_trial = study.best_trial

    # Affichage des résultats principaux
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
