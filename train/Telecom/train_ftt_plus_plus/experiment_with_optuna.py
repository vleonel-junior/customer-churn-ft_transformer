import os
import json
import numpy as np
import optuna
import logging
import gc
import torch

from data.process_telecom_data import get_data
from ftt_plus_plus import FTTPlusPlusConfig, FeatureMapping, FTTPlusPlusPipeline
from train_func import train, val, evaluate, create_loaders

# Configuration du logging (affiche uniquement les erreurs importantes)
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# --- Paramètres fixes ---
seeds = [0, 1, 2]
metrics_dir = "results/results_telecom/ftt_plus_plus_optuna/"
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
    # Hyperparamètres séparés pour chaque étape
    d_token_stage1 = trial.suggest_categorical("d_token_stage1", [16, 32, 64, 128])
    n_blocks_stage1 = trial.suggest_int("n_blocks_stage1", 1, 6)
    ffn_hidden_stage1 = trial.suggest_categorical("ffn_hidden_stage1", [64, 128, 256])
    attention_dropout_stage1 = trial.suggest_float("attention_dropout_stage1", 0.0, 0.3)
    ffn_dropout_stage1 = trial.suggest_float("ffn_dropout_stage1", 0.0, 0.3)
    residual_dropout_stage1 = trial.suggest_float("residual_dropout_stage1", 0.0, 0.2)
    lr_stage1 = trial.suggest_float("lr_stage1", 1e-5, 1e-1, log=True)
    weight_decay_stage1 = trial.suggest_float("weight_decay_stage1", 1e-6, 1e-1, log=True)

    d_token_stage2 = trial.suggest_categorical("d_token_stage2", [16, 32, 64, 128])
    n_blocks_stage2 = trial.suggest_int("n_blocks_stage2", 1, 6)
    ffn_hidden_stage2 = trial.suggest_categorical("ffn_hidden_stage2", [64, 128, 256])
    attention_dropout_stage2 = trial.suggest_float("attention_dropout_stage2", 0.0, 0.3)
    ffn_dropout_stage2 = trial.suggest_float("ffn_dropout_stage2", 0.0, 0.3)
    residual_dropout_stage2 = trial.suggest_float("residual_dropout_stage2", 0.0, 0.2)
    lr_stage2 = trial.suggest_float("lr_stage2", 1e-5, 1e-1, log=True)
    weight_decay_stage2 = trial.suggest_float("weight_decay_stage2", 1e-6, 1e-1, log=True)

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    patience = trial.suggest_int("patience", 5, 30)
    embedding_type = trial.suggest_categorical("embedding_type", [
        "L", "LR", "LR-LR", "Q", "Q-L", "Q-LR", "Q-LR-LR", "T", "T-L", "T-LR", "T-LR-LR", "P", "P-L", "P-LR", "P-LR-LR"
    ])
    M = trial.suggest_int("M", 5, 20)
    k = trial.suggest_int("k", 2, 10)

    aucs = []
    pr_aucs = []
    all_seed_results = []

    for seed_idx, seed_val in enumerate(seeds):
        # Préparation des données
        X, y, cat_cardinalities = get_data(seed_val)

        # Mapping des features (adapter si besoin)
        feature_mapping = FeatureMapping.create_mapping(
            num_feature_names=['tenure', 'MonthlyCharges', 'TotalCharges'],
            cat_feature_names=[
                'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod'
            ]
        )

        # Construction de la config FTT++ avec hyperparamètres séparés
        ftt_plus_config = {
            'd_token': d_token_stage1,
            'n_blocks': n_blocks_stage1,
            'attention_dropout': attention_dropout_stage1,
            'ffn_d_hidden': ffn_hidden_stage1,
            'ffn_dropout': ffn_dropout_stage1,
            'residual_dropout': residual_dropout_stage1,
            'd_out': 1,
            'lr': lr_stage1,
            'weight_decay': weight_decay_stage1
        }
        random_model_config = {
            'd_token': d_token_stage2,
            'n_blocks': n_blocks_stage2,
            'attention_dropout': attention_dropout_stage2,
            'ffn_d_hidden': ffn_hidden_stage2,
            'ffn_dropout': ffn_dropout_stage2,
            'residual_dropout': residual_dropout_stage2,
            'd_out': 1,
            'lr': lr_stage2,
            'weight_decay': weight_decay_stage2
        }

        # Correction : définir un dossier de résultats unique pour chaque trial/seed
        output_dir = os.path.join(metrics_dir, f"trial_{trial.number}", f"seed_{seed_val}")
        os.makedirs(output_dir, exist_ok=True)
        config = FTTPlusPlusConfig(
            ftt_plus_config=ftt_plus_config,
            M=M,
            k=k,
            random_model_config=random_model_config,
            attention_seed=seed_val,
            results_dir=output_dir,
            save_intermediate=False
        )

        # Pipeline FTT++ complet
        pipeline = FTTPlusPlusPipeline(config, feature_mapping)

        # Entraînement pipeline complet
        results = pipeline.run_complete_pipeline(
            X=X,
            y=y,
            cat_cardinalities=cat_cardinalities,
            train_func=train,
            val_func=val,
            evaluate_func=evaluate,
            create_loaders_func=create_loaders,
            stage1_epochs=100,
            stage2_epochs=100,
            batch_size=batch_size,
            patience=patience,
            seed=seed_val,
            embedding_type=embedding_type,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Récupérer la métrique principale (AUC)
        test_performance = results.get("test_performance", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        val_performance = results.get("val_performance", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        aucs.append(test_performance[0] if isinstance(test_performance, (list, tuple)) else test_performance.get("roc_auc", 0))
        pr_aucs.append(test_performance[1] if isinstance(test_performance, (list, tuple)) else test_performance.get("pr_auc", 0))

        # Sauvegarde détaillée par seed/trial
        output_dir = os.path.join(metrics_dir, f"trial_{trial.number}", f"seed_{seed_val}")
        os.makedirs(f"{output_dir}/best_models", exist_ok=True)
        os.makedirs(f"{output_dir}/métriques", exist_ok=True)
        result = {
            "trial_number": trial.number,
            "seed": seed_val,
            # Stage 1
            "d_token_stage1": d_token_stage1,
            "n_blocks_stage1": n_blocks_stage1,
            "ffn_hidden_stage1": ffn_hidden_stage1,
            "attention_dropout_stage1": attention_dropout_stage1,
            "ffn_dropout_stage1": ffn_dropout_stage1,
            "residual_dropout_stage1": residual_dropout_stage1,
            "lr_stage1": lr_stage1,
            "weight_decay_stage1": weight_decay_stage1,
            # Stage 2
            "d_token_stage2": d_token_stage2,
            "n_blocks_stage2": n_blocks_stage2,
            "ffn_hidden_stage2": ffn_hidden_stage2,
            "attention_dropout_stage2": attention_dropout_stage2,
            "ffn_dropout_stage2": ffn_dropout_stage2,
            "residual_dropout_stage2": residual_dropout_stage2,
            "lr_stage2": lr_stage2,
            "weight_decay_stage2": weight_decay_stage2,
            "batch_size": batch_size,
            "patience": patience,
            "M": M,
            "k": k,
            "embedding_type": embedding_type,
            "best_val_loss": results.get("best_val_loss", None),
            "test_performance": to_named_dict(test_performance),
            "val_performance": to_named_dict(val_performance)
        }
        all_seed_results.append(result)
        with open(f'{output_dir}/métriques/ftt_training_results.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Nettoyage mémoire explicite
        torch.cuda.empty_cache()
        gc.collect()

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    mean_pr_auc = np.mean(pr_aucs)

    trial.set_user_attr("detailed_results", {
        "hyperparams": {
            # Stage 1
            "d_token_stage1": d_token_stage1,
            "n_blocks_stage1": n_blocks_stage1,
            "ffn_hidden_stage1": ffn_hidden_stage1,
            "attention_dropout_stage1": attention_dropout_stage1,
            "ffn_dropout_stage1": ffn_dropout_stage1,
            "residual_dropout_stage1": residual_dropout_stage1,
            "lr_stage1": lr_stage1,
            "weight_decay_stage1": weight_decay_stage1,
            # Stage 2
            "d_token_stage2": d_token_stage2,
            "n_blocks_stage2": n_blocks_stage2,
            "ffn_hidden_stage2": ffn_hidden_stage2,
            "attention_dropout_stage2": attention_dropout_stage2,
            "ffn_dropout_stage2": ffn_dropout_stage2,
            "residual_dropout_stage2": residual_dropout_stage2,
            "lr_stage2": lr_stage2,
            "weight_decay_stage2": weight_decay_stage2,
            "batch_size": batch_size,
            "patience": patience,
            "M": M,
            "k": k,
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

    # Sauvegarde intermédiaire globale
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
        study_name="ftt_plus_plus_optuna_enhanced",
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
        logger.info("Optimization interrupted by user")

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

    logger.info(f"Optimization completed!")
    logger.info(f"Best trial: {best_trial.number}")
    logger.info(f"Best mean AUC: {best_trial.value:.4f}")
    logger.info(f"Best params: {best_trial.params}")

    if len(study.trials) > 10:
        importance = optuna.importance.get_param_importances(study)
        logger.info("Parameter importance:")
        for param, imp in importance.items():
            logger.info(f"  {param}: {imp:.4f}")

        with open(os.path.join(metrics_dir, "param_importance.json"), "w") as f:
            json.dump(importance, f, indent=2)
