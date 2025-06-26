from utils import performance
import zero
import torch
import numpy as np

def apply_model(model, x_num, x_cat):
    """
    Fonction qui applique le modèle avec les données numériques et catégorielles
    """
    return model(x_num, x_cat)

sigmoid = torch.nn.Sigmoid()

@torch.no_grad()
def evaluate(model, part, X, y, seed):
    model.eval()
    prediction = []
    x_num, x_cat = X[part]
    
    for batch_idx in zero.iter_batches(range(int(y[part].size(0))), 1024):
        # CORRECTION : Convertir batch_idx en liste d'entiers Python
        if isinstance(batch_idx, torch.Tensor):
            batch_idx = batch_idx.cpu().numpy().tolist()
        
        batch_x_num = x_num[batch_idx]
        batch_x_cat = x_cat[batch_idx]
        # Appliquer sigmoid seulement pour l'évaluation (pas pendant l'entraînement)
        output = sigmoid(apply_model(model, batch_x_num, batch_x_cat).squeeze(1))
        prediction.append(output)
    
    prediction = torch.cat(prediction).cpu().numpy()
    target = y[part].cpu().numpy()
    
    roc_auc, pr_auc, acc, ba, mcc, sensitivity, specificity, precision, f1, ck = performance(
        target, prediction, threshold=0.5
    )
    test_performance = [roc_auc, pr_auc, acc, ba, mcc, sensitivity, specificity, precision, f1, ck]
    return test_performance

def train(epoch, model, optimizer, X, y, train_loader, loss_fn):
    model.train()
    loss_train = 0
    x_num, x_cat = X['train']
    
    for iteration, batch_idx in enumerate(train_loader):
        optimizer.zero_grad()
        
        batch_x_num = x_num[batch_idx]
        batch_x_cat = x_cat[batch_idx]
        y_batch = y['train'][batch_idx].float()
        
        # IMPORTANT: Passer les deux arguments séparément
        output = apply_model(model, batch_x_num, batch_x_cat).squeeze(1)
        loss = loss_fn(output, y_batch)
        
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
    
    loss_train = loss_train / len(train_loader)
    print(f'Epoch {epoch:03d} | Training loss: {loss_train:.4f}')
    return loss_train

def val(epoch, model, X, y, val_loader, loss_fn):
    model.eval()
    loss_val = 0
    x_num, x_cat = X['val']
    
    with torch.no_grad():  # Ajout du no_grad pour la validation
        for iteration, batch_idx in enumerate(val_loader):
            batch_x_num = x_num[batch_idx]
            batch_x_cat = x_cat[batch_idx]
            y_batch = y['val'][batch_idx].float()
            
            # IMPORTANT: Passer les deux arguments séparément
            output = apply_model(model, batch_x_num, batch_x_cat).squeeze(1)
            loss = loss_fn(output, y_batch)
            loss_val += loss.item()
    
    loss_val = loss_val / len(val_loader)
    print(f'Epoch {epoch:03d} | Validation loss: {loss_val:.4f}')
    return loss_val