import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.preprocessing
import torch
import warnings

# Supprimer les warnings sklearn
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHURN_XLS_PATH = "data/Bank/Churn_Modelling.xls"

def read_data(path):
    df = pd.read_excel(path)

    # Colonnes à supprimer
    df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

    # Colonnes numériques et catégorielles
    num_cols = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
        'EstimatedSalary'
    ]
    # Les variables binaires HasCrCard et IsActiveMember sont traitées comme catégorielles
    cat_cols = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
    target_col = 'Exited'

    print(f"Données originales: {df.shape[0]} lignes")

    # Nettoyage : suppression des lignes avec NaN
    df_clean = df.dropna(subset=num_cols + cat_cols + [target_col])
    print(f"Après nettoyage: {df_clean.shape[0]} lignes ({df.shape[0] - df_clean.shape[0]} supprimées)")

    # Extraction des features numériques
    X_num = df_clean[num_cols].to_numpy().astype('float32')

    # Encodage des variables catégorielles en indices entiers (OrdinalEncoder)
    ord_enc = sklearn.preprocessing.OrdinalEncoder()
    X_cat = ord_enc.fit_transform(df_clean[cat_cols]).astype('int64')
    cat_cardinalities = [len(categories) for categories in ord_enc.categories_]

    print(f"\nInformations sur les features:")
    print(f"Features numériques ({len(num_cols)}): {num_cols}")
    print(f"Variables catégorielles ({len(cat_cols)}): {cat_cols}")
    print(f"Cardinalités des variables catégorielles: {cat_cardinalities}")

    # Encodage de la cible
    y_all = df_clean[target_col].astype('int64').to_numpy()

    print(f"Distribution des classes: {np.bincount(y_all)}")

    return X_num, X_cat, y_all, cat_cardinalities

def get_data(seed):
    X_num, X_cat, y_all, cat_cardinalities = read_data(CHURN_XLS_PATH)

    # Premier split: train+val vs test (80/20)
    X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X_num, X_cat, y_all, train_size=0.8, random_state=seed, stratify=y_all
    )

    # Deuxième split: train vs val (85/15 du train+val)
    X_num_train, X_num_val, X_cat_train, X_cat_val, y_train, y_val = sklearn.model_selection.train_test_split(
        X_num_train, X_cat_train, y_train, train_size=0.85, random_state=seed, stratify=y_train
    )

    print(f"Tailles des ensembles:")
    print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    # Standardisation des features numériques
    preprocess = sklearn.preprocessing.StandardScaler().fit(X_num_train)
    X_num_train = torch.tensor(preprocess.transform(X_num_train), device=device)
    X_num_val = torch.tensor(preprocess.transform(X_num_val), device=device)
    X_num_test = torch.tensor(preprocess.transform(X_num_test), device=device)

    # Conversion en tensors PyTorch
    X_cat_train = torch.tensor(X_cat_train, device=device)
    X_cat_val = torch.tensor(X_cat_val, device=device)
    X_cat_test = torch.tensor(X_cat_test, device=device)

    y_train = torch.tensor(y_train, device=device)
    y_val = torch.tensor(y_val, device=device)
    y_test = torch.tensor(y_test, device=device)

    # Organisation des données
    X = {
        'train': (X_num_train, X_cat_train),
        'val': (X_num_val, X_cat_val),
        'test': (X_num_test, X_cat_test)
    }
    y = {
        'train': y_train,
        'val': y_val,
        'test': y_test
    }

    return X, y, cat_cardinalities
