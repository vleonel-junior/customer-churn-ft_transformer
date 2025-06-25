from utils import performance
import zero
import torch
import numpy as np
def apply_model(model, x_num, x_cat=None):
    return model(x_num, x_cat)

sigmoid = torch.nn.Sigmoid()
@torch.no_grad()
def evaluate(model, part, X, y, seed):
    model.eval()
    prediction = []
    for batch in zero.iter_batches(X[part], 1024):
        output = sigmoid(apply_model(model, batch).squeeze(1))
        prediction.append(output)
    prediction = torch.cat(prediction).cpu().numpy()
    target = y[part].cpu().numpy()

    # np.save(f'./outputs/seed_{seed}/probs.npy', prediction)
    # np.save(f'./outputs/seed_{seed}/labels.npy', target)

    roc_auc, pr_auc, acc, ba, mcc, sensitivity, specificity, precision, f1, ck = performance(target, prediction, thresold=0.5)

    test_performace = [roc_auc, pr_auc, acc, ba, mcc, sensitivity, specificity, precision, f1, ck]
    return test_performace


def train(epoch, model, optimizer,X, y, train_loader, loss_fn):
    model.train()
    loss_train = 0
    for iteration, batch_idx in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        x_batch = X['train'][batch_idx]
        y_batch = y['train'][batch_idx].float()
        output = sigmoid(apply_model(model, x_batch).squeeze(1))
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()
        # if iteration % report_frequency == 0:
        #     print(f'(epoch) {epoch} (batch) {iteration} (loss) {loss.item():.4f}')
        loss_train += loss.item()
    loss_train = loss_train/len(train_loader)
    print(f'Epoch {epoch:03d} | Training loss: {loss_train:.4f}')
    return loss_train

def val(epoch, model, X, y, val_loader, loss_fn):
    model.eval()
    loss_val = 0
    for iteration, batch_idx in enumerate(val_loader):
        x_batch = X['val'][batch_idx]
        y_batch = y['val'][batch_idx].float()
        output = sigmoid(apply_model(model, x_batch).squeeze(1))
        loss = loss_fn(output, y_batch)
        loss_val += loss.item()
    loss_val = loss_val/len(val_loader)
    print(f'Epoch {epoch:03d} | Validation loss: {loss_val:.4f}')
    return loss_val