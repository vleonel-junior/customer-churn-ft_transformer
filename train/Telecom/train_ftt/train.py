import rtdl
import zero
from data.process_telecom_data import device, get_data
from train_funct import train, val, evaluate
import numpy as np 
import time 
import torch
import os

if __name__ == '__main__':
    # Paramètres
    d_out = 1
    lr = 0.001
    weight_decay = 0.0
    batch_size = 64
    n_epochs = 50
    seed = 0
    patience = 10  # Early stopping
    
    # NOUVEAU: Dimension d'embedding cohérente
    d_embedding = 192  # Dimension standard pour FT-Transformer
    
    # Créer le dossier de sortie si nécessaire
    output_dir = f'./outputs/seed_{seed}'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Utilisation du device: {device}")
    print(f"Seed: {seed}")
    print(f"Dimension d'embedding: {d_embedding}")
    
    # Charger les données
    X, y, cat_cardinalities = get_data(seed)
    
    # Créer les loaders
    train_loader = zero.data.IndexLoader(len(y['train']), batch_size, device=device)
    val_loader = zero.data.IndexLoader(len(y['val']), batch_size, device=device)
    
    # Modèle avec dimensions cohérentes
    from num_embedding_factory import get_num_embedding

    # CORRECTION: Embedding numérique avec la même dimension
    num_embedding = get_num_embedding(
        "LR",  # ou "linear" selon votre implémentation
        X['train'][0],
        d_embedding=d_embedding  # Même dimension que le modèle
    )

    # CORRECTION: Spécifier d_embedding explicitement
    model = rtdl.FTTransformer.make_default(
        n_num_features=X['train'][0].shape[1],
        cat_cardinalities=cat_cardinalities,
        d_embedding=d_embedding,  # Dimension explicite
        last_layer_query_idx=[-1],  # Optimisation
        d_out=d_out,
    )
    
    # Remplacer l'embedding numérique par défaut
    model.feature_tokenizer.num_tokenizer = num_embedding
    model.to(device)
    
    # Optimiseur
    optimizer = (
        model.make_default_optimizer()
        if isinstance(model, rtdl.FTTransformer)
        else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    )
    
    # Fonction de perte
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    print(f"Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    # VERIFICATION: Dimensions des embeddings
    print(f"Features numériques: {X['train'][0].shape[1]}")
    print(f"Features catégorielles: {len(cat_cardinalities)}")
    print(f"Cardinalités: {cat_cardinalities}")
    
    # Test rapide des dimensions
    with torch.no_grad():
        sample_x_num = torch.tensor(X['train'][0][:1], dtype=torch.float32, device=device)
        sample_x_cat = torch.tensor(X['train'][1][:1], dtype=torch.long, device=device)
        try:
            _ = model.feature_tokenizer(sample_x_num, sample_x_cat)
            print("✓ Test des dimensions réussi")
        except Exception as e:
            print(f"❌ Erreur de dimensions: {e}")
            exit(1)
    
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
    
    # Sauvegarde des résultats
    results = {
        'train_losses': train_loss_list,
        'val_losses': val_loss_list,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'test_performance': test_performance,
        'val_performance': val_performance,
        'config': {
            'd_embedding': d_embedding,
            'lr': lr,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'seed': seed
        }
    }
    
    np.save(f'{output_dir}/training_results.npy', results)
    torch.save(model.state_dict(), f'{output_dir}/best_model.pt')
    
    print(f"\nRésultats sauvegardés dans {output_dir}/")
    print("Entraînement terminé!")