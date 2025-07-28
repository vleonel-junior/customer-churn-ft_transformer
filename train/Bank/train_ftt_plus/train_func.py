from utils import performance
import zero
import torch
import numpy as np

def apply_model(model, x_num, x_cat=None):
    """Applique le modèle aux données d'entrée - retourne seulement les logits"""
    return model(x_num, x_cat, return_attention=False)

@torch.no_grad()
def evaluate(model, part, X, y, seed):
    """Évalue le modèle sur un ensemble de données"""
    model.eval()
    prediction = []
    
    for batch in zero.iter_batches(X[part], 1024):
        x_num_batch, x_cat_batch = batch
        # Pas de sigmoid ici car BCEWithLogitsLoss l'applique déjà
        output = apply_model(model, x_num_batch, x_cat_batch).squeeze(1)
        # Appliquer sigmoid seulement pour les prédictions finales
        prediction.append(torch.sigmoid(output))
    
    prediction = torch.cat(prediction).cpu().numpy()
    target = y[part].cpu().numpy()
    
    # Optionnel: sauvegarder les prédictions
    # np.save(f'./outputs/seed_{seed}/probs.npy', prediction)
    # np.save(f'./outputs/seed_{seed}/labels.npy', target)
    
    roc_auc, pr_auc, acc, ba, mcc, sensitivity, specificity, precision, f1, ck = performance(
        target, prediction, thresold=0.5
    )
    test_performance = [roc_auc, pr_auc, acc, ba, mcc, sensitivity, specificity, precision, f1, ck]
    
    print(f'{part.capitalize()} Performance:')
    print(f'  ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Accuracy: {acc:.4f}')
    print(f'  F1: {f1:.4f} | MCC: {mcc:.4f} | Balanced Acc: {ba:.4f}')
    
    return test_performance

def train(epoch, model, optimizer, X, y, train_loader, loss_fn):
    """Entraîne le modèle pour une époque"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for iteration, batch_idx in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Récupérer les données du batch
        x_num_batch = X['train'][0][batch_idx]
        x_cat_batch = X['train'][1][batch_idx]
        y_batch = y['train'][batch_idx].float()
        
        # Forward pass - PAS de sigmoid car BCEWithLogitsLoss l'applique
        output = apply_model(model, x_num_batch, x_cat_batch).squeeze(1)
        
        # Calcul de la perte
        loss = loss_fn(output, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Affichage optionnel du progrès (commenté pour réduire le bruit)
        # if iteration % 50 == 0:
        #     print(f'  Batch {iteration:3d} | Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / num_batches
    print(f'Epoch {epoch:03d} | Training loss: {avg_loss:.4f}')
    return avg_loss

def val(epoch, model, X, y, val_loader, loss_fn):
    """Valide le modèle pour une époque"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():  # Important pour la validation
        for iteration, batch_idx in enumerate(val_loader):
            # Récupérer les données du batch
            x_num_batch = X['val'][0][batch_idx]
            x_cat_batch = X['val'][1][batch_idx]
            y_batch = y['val'][batch_idx].float()
            
            # Forward pass - PAS de sigmoid car BCEWithLogitsLoss l'applique
            output = apply_model(model, x_num_batch, x_cat_batch).squeeze(1)
            
            # Calcul de la perte
            loss = loss_fn(output, y_batch)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    print(f'Epoch {epoch:03d} | Validation loss: {avg_loss:.4f}')
    return avg_loss