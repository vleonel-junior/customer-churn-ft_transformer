import zero
import torch
import numpy as np
import time
import os
from data.process_bank_data import device, get_data
from train_func import train, val, evaluate
from sparse_ftt_plus.model import InterpretableFTTPlus
from interpretability_analyzer import analyze_interpretability
from num_embedding_factory import get_num_embedding

if __name__ == '__main__':
    # Paramètres
    d_out = 1
    lr = 1.0540647524918737e-05
    weight_decay = 0.0003360870237649223
    batch_size = 32
    n_epochs = 100
    seed = 0
    patience = 29  # Early stopping

    # Créer le dossier de sortie si nécessaire
    output_dir = f'results/results_bank/sparse_ftt_plus/seed_{seed}'
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

    # Configuration du modèle Sparse FTT+
    n_num_features = X['train'][0].shape[1]
    d_token = 128
    
    print(f"Configuration du modèle:")
    print(f"  - Features numériques: {n_num_features}")
    print(f"  - Features catégorielles: {len(cat_cardinalities)} (cardinalités: {cat_cardinalities})")
    print(f"  - Taille des tokens: {d_token}")

    # Créer le modèle
    model = InterpretableFTTPlus.make_baseline(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        d_token=128,
        n_blocks=3,
        n_heads=16,
        attention_dropout=0.2473988634060151,
        ffn_d_hidden=256,
        ffn_dropout=0.17474890937885124,
        residual_dropout=0.12087417161076972,
        d_out=d_out
    )

    # Embedding numérique personnalisé (optionnel)
    embedding_type = "T"
    print(f"Type d'embedding numérique: {embedding_type}")

    # Forcer les tenseurs sur CPU pour éviter le warning rtdl_num_embeddings
    X_train_cpu = X['train'][0].cpu()
    num_embedding = get_num_embedding(
        embedding_type=embedding_type,
        X_train=X_train_cpu,
        d_embedding=d_token,
        y_train=y['train'] if embedding_type in ("T", "T-L", "T-LR", "T-LR-LR") else None
    )
    # Assigner l'embedding numérique au modèle
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
    feature_names = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
        'Geography', 'Gender', 'HasCrCard', 'IsActiveMember'
    ]
    
    analyze_interpretability(
        model=model, X=X, y=y, model_name='interpretable_sparse_ftt_plus', seed=seed,
        model_config={
            'n_num_features': n_num_features, 
            'cat_cardinalities': cat_cardinalities,
            'd_token': d_token, 
            'n_blocks': 3, 
            'n_heads': 8,
            'attention_dropout': 0.2,
            'ffn_d_hidden': d_token * 4,
            'ffn_dropout': 0.1, 
            'residual_dropout': 0.0, 
            'embedding_type': embedding_type
        },
        training_results={'train_losses': train_loss_list, 'val_losses': val_loss_list,
                         'best_epoch': best_epoch, 'best_val_loss': best_val_loss},
        performance_results={'val': val_performance, 'test': test_performance},
        feature_names=feature_names,
        local_output_dir=output_dir,
        results_base_dir=output_dir
    )

    print("Entraînement terminé!")