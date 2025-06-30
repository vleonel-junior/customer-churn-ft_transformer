import torch
import numpy as np
from num_embedding_factory import get_num_embedding

train_batch_size = 32
test_batch_size = 16
n_features = 6
d_embedding = 8
n_bins = 4  # Pour les embeddings binning

embedding_types = [
    "L", "LR", "LR-LR",
    "Q", "Q-L", "Q-LR", "Q-LR-LR",
    "T", "T-L", "T-LR", "T-LR-LR",
    "P", "P-L", "P-LR", "P-LR-LR"
]

for embedding_type in embedding_types:
    print("="*60)
    print(f"Test embedding type: {embedding_type}")

    # Pour la phase d'initialisation (fit), il FAUT un dataset "suffisant" pour Q/Quantile
    if embedding_type.startswith("Q"):
        X_train = np.random.randn(500, n_features)  # dataset plus gros pour quantile fitting
    else:
        X_train = np.random.randn(train_batch_size, n_features)

    y_train = np.random.randint(0, 2, X_train.shape[0])
    X_batch_test = np.random.randn(test_batch_size, n_features)
    X_batch_test_tensor = torch.FloatTensor(X_batch_test)

    params = {
        "embedding_type": embedding_type,
        "X_train": X_train,
        "d_embedding": d_embedding,
    }

    if embedding_type.startswith("T"):
        params["y_train"] = y_train
        params["n_bins"] = n_bins
    elif embedding_type.startswith("Q"):
        params["n_bins"] = n_bins
    elif embedding_type.startswith("P"):
        params["d_periodic_embedding"] = 6
    elif embedding_type.startswith("L"):
        params["n_bins"] = n_bins

    embedder = get_num_embedding(**params)
    with torch.no_grad():
        out = embedder(X_batch_test_tensor)
    print("X_batch_test_tensor shape:", X_batch_test_tensor.shape)
    print("Sortie shape:", out.shape)
    assert out.shape == (test_batch_size, n_features, d_embedding), \
        f"Shape attendue: ({test_batch_size}, {n_features}, {d_embedding}), obtenu: {out.shape}"
    print("✅ Test PASSED\n")