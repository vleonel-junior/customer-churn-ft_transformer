import rtdl
import zero
from data.process_telecom_data import device, get_data
from train_funct import train, val, evaluate
import numpy as np
import time
import torch
import copy

if __name__ == '__main__':
    # Params
    d_out = 1
    lr = 0.001
    weight_decay = 0.0
    batch_size = 64
    n_epochs = 50
    seed = 0
    patience = 10  # Pour early stopping
    
    # Load data
    X, y, cat_cardinalities = get_data(seed)
    train_loader = zero.data.IndexLoader(len(y['train']), batch_size, device=device)
    val_loader = zero.data.IndexLoader(len(y['val']), batch_size, device=device)
    
    # Model
    model = rtdl.FTTransformer.make_default(
        n_num_features=X['train'][0].shape[1],
        cat_cardinalities=cat_cardinalities,
        last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
        d_out=d_out,
    )
    model.to(device)
    
    optimizer = (
        model.make_default_optimizer()
        if isinstance(model, rtdl.FTTransformer)
        else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    )
    
    # Loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    # Training avec early stopping amélioré
    train_loss_list = []
    val_loss_list = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        
        # Training
        loss_train = train(epoch, model, optimizer, X, y, train_loader, loss_fn)
        
        # Validation
        loss_val = val(epoch, model, X, y, val_loader, loss_fn)
        
        train_loss_list.append(loss_train)
        val_loss_list.append(loss_val)
        
        # Early stopping et sauvegarde du meilleur modèle
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f' <<< BEST VALIDATION EPOCH (loss: {loss_val:.4f})')
        else:
            patience_counter += 1
        
        epoch_time = time.time() - epoch_start
        print(f'Epoch time: {epoch_time:.2f}s')
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping after {epoch + 1} epochs')
            break
    
    # Charger le meilleur modèle pour l'évaluation finale
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Loaded best model with validation loss: {best_val_loss:.4f}')
    
    # Évaluation finale sur le test set (seulement à la fin)
    print("\nEvaluating on test set...")
    test_start = time.time()
    test_performance = evaluate(model, 'test', X, y, seed)
    test_time = time.time() - test_start
    
    total_time = time.time() - start_time
    
    # Affichage des résultats
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Training completed in {total_time:.2f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Test evaluation time: {test_time:.2f}s")
    
    metrics_names = ['ROC-AUC', 'PR-AUC', 'Accuracy', 'Balanced Accuracy', 
                     'MCC', 'Sensitivity', 'Specificity', 'Precision', 'F1', 'Cohen Kappa']
    
    print("\nTest Performance:")
    for name, score in zip(metrics_names, test_performance):
        print(f"{name:>18}: {score:.4f}")
    
    # Optionnel : sauvegarder le modèle et les métriques
    # torch.save(best_model_state, f'best_model_seed_{seed}.pth')
    # np.save(f'train_losses_seed_{seed}.npy', train_loss_list)
    # np.save(f'val_losses_seed_{seed}.npy', val_loss_list)
    # np.save(f'test_performance_seed_{seed}.npy', test_performance)