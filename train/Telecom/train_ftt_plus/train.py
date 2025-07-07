import zero
import torch
import numpy as np
import time
import os
from data.process_telecom_data import device, get_data
from train_func import train, val, evaluate
from ftt_plus.model import InterpretableFTTPlus
from interpretability_analyzer import analyze_interpretability
from num_embedding_factory import get_num_embedding

if __name__ == '__main__':
    # Paramètres
    d_out = 1
    lr = 0.001
    weight_decay = 0.0
    batch_size = 64
    n_epochs = 100
    seed = 0
    patience = 20  # Early stopping

    # Créer le dossier de sortie si nécessaire
    output_dir = f'results/results_telecom/ftt_plus/seed_{seed}'
    os.makedirs(f"{output_dir}/heatmaps", exist_ok=True)
    os.makedirs(f"{output_dir}/best_models", exist_ok=True)
    os.makedirs(f"{output_dir}/métriques", exist_ok=True)

    print(f"Utilisation du device: {device}")
    print(f"Seed: {seed}")

    # Charger les données
    X, y, cat_cardinalities = get_data(seed)

    # Créer les loaders
    train_loader = zero.data.IndexLoader(len(y['train']), batch_size, device=device)
    val_loader = zero.data.IndexLoader(len(y['val']), batch_size, device=device)

    # Configuration du modèle FTT+ refactorisé
    n_num_features = X['train'][0].shape[1]
    d_token = 64
    
    print(f"Configuration du modèle:")
    print(f"  - Features numériques: {n_num_features}")
    print(f"  - Features catégorielles: {len(cat_cardinalities)} (cardinalités: {cat_cardinalities})")
    print(f"  - Taille des tokens: {d_token}")

    # Création du modèle avec la nouvelle architecture
    model = InterpretableFTTPlus.make_baseline(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        d_token=d_token,
        n_blocks=2,
        attention_dropout=0.1,
        ffn_d_hidden=128,
        ffn_dropout=0.1,
        residual_dropout=0.1,
        d_out=d_out
    )

    # Embedding numérique personnalisé (optionnel)
    embedding_type = "P-LR-LR"
    print(f"Type d'embedding numérique: {embedding_type}")

    num_embedding = get_num_embedding(
        embedding_type=embedding_type,
        X_train=X['train'][0],
        d_embedding=d_token,
        y_train=y['train'] if embedding_type in ("T", "T-L", "T-LR", "T-LR-LR") else None
    )
    # Remplacer l'embedding numérique par défaut
    model.feature_tokenizer.num_tokenizer = num_embedding

    model.to(device)

    # Optimiseur avec groupes de paramètres optimisés
    optimizer = torch.optim.AdamW(model.optimization_param_groups(), lr=lr, weight_decay=weight_decay)

    # Fonction de perte
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print(f"Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")

    # Entraînement
    train_loss_list = []
    val_loss_list = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    print("\n=== Début de l'entraînement ===")

    for epoch in range(n_epochs):
        start_time = time.time()

        # Entraînement
        train_loss = train(epoch, model, optimizer, X, y, train_loader, loss_fn)

        # Validation
        val_loss = val(epoch, model, X, y, val_loader, loss_fn)

        # Sauvegarde des métriques
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        epoch_time = time.time() - start_time
        print(f'Epoch {epoch:03d} completed in {epoch_time:.2f}s')

        # Early stopping et sauvegarde du meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f' <<< NOUVEAU MEILLEUR MODÈLE (val_loss: {val_loss:.4f})')

            # Évaluation sur l'ensemble de test
            print(' >>> Évaluation sur l\'ensemble de test:')
            test_performance = evaluate(model, 'test', X, y, seed)

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping à l\'époque {epoch} (patience: {patience})')
                break

        print('-' * 60)

    # Charger le meilleur modèle
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nMeilleur modèle chargé (époque {best_epoch}, val_loss: {best_val_loss:.4f})")

    # Évaluation finale
    print("\n=== Évaluation finale ===")
    print("Performance sur l'ensemble de validation:")
    val_performance = evaluate(model, 'val', X, y, seed)

    print("\nPerformance sur l'ensemble de test:")
    test_performance = evaluate(model, 'test', X, y, seed)

    # Analyse d'interprétabilité automatique
    feature_names = ['tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                     'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                     'PaperlessBilling', 'PaymentMethod']
    
    analyze_interpretability(
        model=model, X=X, y=y, model_name='interpretable_ftt_plus', seed=seed,
        model_config={'n_num_features': n_num_features, 'cat_cardinalities': cat_cardinalities,
                     'd_token': d_token, 'n_blocks': 2, 'attention_dropout': 0.1,
                     'ffn_d_hidden': 128, 'ffn_dropout': 0.1, 'residual_dropout': 0.1, 'embedding_type': embedding_type},
        training_results={'train_losses': train_loss_list, 'val_losses': val_loss_list,
                         'best_epoch': best_epoch, 'best_val_loss': best_val_loss},
        performance_results={'val': val_performance, 'test': test_performance},
        feature_names=feature_names,
        local_output_dir=output_dir,
        results_base_dir=output_dir
    )


    print("Entraînement terminé!")