from rtdl_lib.nn._embeddings import (
    LinearEmbeddings,
    make_lr_embeddings,
    make_ple_lr_embeddings,
    make_plr_embeddings,
    PeriodicEmbeddings,
    PiecewiseLinearEncoder,
    NLinear,
)
from rtdl_lib.data import (
    compute_quantile_bin_edges,
    compute_decision_tree_bin_edges,
)
import torch.nn as nn
import torch
import numpy as np


def get_num_embedding(
    embedding_type: str,
    X_train,
    d_embedding: int,
    y_train=None,
    n_bins: int = 5,
    d_periodic_embedding: int = None,
    sigma: float = 0.1,
):
    """
    Retourne un module d'embedding numérique dont la sortie est toujours (batch_size, n_features, d_embedding).

    embedding_type:
        L         : LinearEmbeddings
        LR        : LinearEmbeddings + ReLU
        LR-LR     : LR puis NLinear + ReLU
        Q         : PLE quantile (avec projection vers d_embedding)
        Q-L       : PLE → NLinear
        Q-LR      : PLE → NLinear → ReLU
        Q-LR-LR   : PLE → NLinear → ReLU → NLinear → ReLU
        T         : PLE arbre (avec projection vers d_embedding)
        T-L       : PLE arbre → NLinear
        T-LR      : PLE arbre → NLinear → ReLU
        T-LR-LR   : PLE arbre → NLinear → ReLU → NLinear → ReLU
        P         : PeriodicEmbeddings (avec projection vers d_embedding)
        P-L       : Periodic → NLinear
        P-LR      : Periodic → NLinear → ReLU
        P-LR-LR   : Periodic → NLinear → ReLU → NLinear → ReLU
    """
    # Conversion numpy → tensor si nécessaire
    if isinstance(X_train, np.ndarray):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    if y_train is not None and isinstance(y_train, np.ndarray):
        y_train = torch.tensor(y_train, dtype=torch.long)
    
    n_features = X_train.shape[1]

    # ------ Linear + ReLU ------
    if embedding_type == "L":
        return LinearEmbeddings(n_features, d_embedding, bias=True)

    if embedding_type == "LR":
        return nn.Sequential(
            LinearEmbeddings(n_features, d_embedding, bias=True),
            nn.ReLU(),
        )

    if embedding_type == "LR-LR":
        # LR outputs (B,F,d_embedding) → NLinear → ReLU
        return nn.Sequential(
            LinearEmbeddings(n_features, d_embedding, bias=True),
            nn.ReLU(),
            NLinear(n_features, d_embedding, d_embedding),
            nn.ReLU(),
        )

    # ------ Quantile PLE ------
    if embedding_type in ("Q", "Q-L", "Q-LR", "Q-LR-LR"):
        edges = compute_quantile_bin_edges(X_train, n_bins=n_bins)
        ple = PiecewiseLinearEncoder(edges, stack=True)
        
        # Tous les cas nécessitent une projection vers d_embedding
        if embedding_type == "Q":
            return nn.Sequential(
                ple,
                NLinear(n_features, ple.d_encoding, d_embedding),
            )
        elif embedding_type == "Q-L":
            return nn.Sequential(
                ple,
                NLinear(n_features, ple.d_encoding, d_embedding),
            )
        elif embedding_type == "Q-LR":
            return nn.Sequential(
                ple,
                NLinear(n_features, ple.d_encoding, d_embedding),
                nn.ReLU(),
            )
        else:  # Q-LR-LR
            return nn.Sequential(
                ple,
                NLinear(n_features, ple.d_encoding, d_embedding),
                nn.ReLU(),
                NLinear(n_features, d_embedding, d_embedding),
                nn.ReLU(),
            )

    # ------ Tree-based PLE ------
    if embedding_type in ("T", "T-L", "T-LR", "T-LR-LR"):
        assert y_train is not None, "y_train requis pour PLE arbre"
        edges = compute_decision_tree_bin_edges(
            X_train, 
            n_bins=n_bins, 
            y=y_train,
            regression=False,
            tree_kwargs={'max_depth': 5, 'min_samples_leaf': 20}
        )
        ple = PiecewiseLinearEncoder(edges, stack=True)
        
        # Tous les cas nécessitent une projection vers d_embedding
        if embedding_type == "T":
            return nn.Sequential(
                ple,
                NLinear(n_features, ple.d_encoding, d_embedding),
            )
        elif embedding_type == "T-L":
            return nn.Sequential(
                ple,
                NLinear(n_features, ple.d_encoding, d_embedding),
            )
        elif embedding_type == "T-LR":
            return nn.Sequential(
                ple,
                NLinear(n_features, ple.d_encoding, d_embedding),
                nn.ReLU(),
            )
        else:  # T-LR-LR
            return nn.Sequential(
                ple,
                NLinear(n_features, ple.d_encoding, d_embedding),
                nn.ReLU(),
                NLinear(n_features, d_embedding, d_embedding),
                nn.ReLU(),
            )

    # ------ Periodic ------
    if embedding_type in ("P", "P-L", "P-LR", "P-LR-LR"):
        if d_periodic_embedding is None:
            d_periodic_embedding = d_embedding
            
        pe = PeriodicEmbeddings(n_features, d_periodic_embedding, sigma)
        
        # Tous les cas nécessitent une projection vers d_embedding
        if embedding_type == "P":
            return nn.Sequential(
                pe,
                NLinear(n_features, d_periodic_embedding, d_embedding),
            )
        elif embedding_type == "P-L":
            return nn.Sequential(
                pe,
                NLinear(n_features, d_periodic_embedding, d_embedding),
            )
        elif embedding_type == "P-LR":
            return nn.Sequential(
                pe,
                NLinear(n_features, d_periodic_embedding, d_embedding),
                nn.ReLU(),
            )
        else:  # P-LR-LR
            return nn.Sequential(
                pe,
                NLinear(n_features, d_periodic_embedding, d_embedding),
                nn.ReLU(),
                NLinear(n_features, d_embedding, d_embedding),
                nn.ReLU(),
            )

    raise ValueError(f"Type d'embedding inconnu : {embedding_type}")


def debug_embedding_dimensions(embedding_module, n_features, batch_size=32):
    """
    Fonction utilitaire pour déboguer les dimensions des embeddings
    """
    # Créer un batch d'exemple
    x_dummy = torch.randn(batch_size, n_features)
    
    with torch.no_grad():
        output = embedding_module(x_dummy)
        print(f"Input shape: {x_dummy.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected format: (batch_size={batch_size}, n_features={n_features}, d_embedding)")
        
        if len(output.shape) == 3:
            batch_dim, feat_dim, emb_dim = output.shape
            print(f"✓ Format correct: batch={batch_dim}, features={feat_dim}, embedding={emb_dim}")
        else:
            print(f"✗ Format incorrect: {output.shape}")
    
    return output.shape