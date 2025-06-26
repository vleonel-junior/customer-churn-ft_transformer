import rtdl
import zero
from data.process_telecom_data import device, get_data
from train_funct import train, val, evaluate
import numpy as np 
import time 
import torch

if __name__ == '__main__':
    # Params
    d_out = 1
    lr = 0.001
    weight_decay = 0.0
    batch_size = 64
    n_epochs = 50
    seed = 0

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
    loss_fn = torch.nn.BCELoss()

    # Training 
    train_loss_list = []
    val_loss_list = []
    val_loss_check = 10
    for epoch in range(n_epochs):
        start = time.time()
        loss_train = train(epoch, model, optimizer, X, y, train_loader, loss_fn)
        loss_val = val(epoch, model, X, y, val_loader, loss_fn)
        train_loss_list.append(loss_train)
        val_loss_list.append(loss_val)
        if loss_val <= val_loss_check:
            print(' <<< BEST VALIDATION EPOCH')
            start = time.time()
            matrix = evaluate(model, 'test', X, y, seed)
            val_loss_check = loss_val
