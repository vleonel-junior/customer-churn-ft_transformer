import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.preprocessing
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_data(path):
    df = pd.read_csv(path)

    # Conversion de 'TotalCharges' en numérique
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Colonnes numériques et catégorielles
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    target_col = 'Churn'

    # Suppression des lignes contenant des NaN dans les colonnes d'intérêt
    df = df.dropna(subset=num_cols + cat_cols + [target_col])

    # Extraction des features numériques
    numerical = df[num_cols].to_numpy()

    # Encodage OneHot séparé pour chaque variable catégorielle
    ohe = sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
    cat_encoded_list = []
    cat_cardinalities = []
    for col in cat_cols:
        encoded = ohe.fit_transform(df[[col]])
        cat_encoded_list.append(encoded)
        cat_cardinalities.append(len(ohe.categories_[0]))
    categorical = np.concatenate(cat_encoded_list, axis=1)

    # Concaténation des features numériques et catégorielles encodées
    encoded_data = np.concatenate((numerical, categorical), axis=-1)

    # Encodage de la cible
    label = df[target_col].map({'Yes': 1, 'No': 0}).astype(int).to_numpy()

    X_all = encoded_data.astype('float32')
    y_all = label.astype('int64')

    return X_all, y_all, cat_cardinalities

def get_data(seed):
    X_all, y_all, cat_cardinalities = read_data('./data/Telco_Customer_Churn.csv')
    X = {}
    y = {}
    X['train'], X['test'], y['train'], y['test'] = sklearn.model_selection.train_test_split(
        X_all, y_all, train_size=0.8, random_state=seed
    )
    X['train'], X['val'], y['train'], y['val'] = sklearn.model_selection.train_test_split(
        X['train'], y['train'], train_size=0.85, random_state=0
    )
    print(len(y['train']), len(y['val']), len(y['test']))

    preprocess = sklearn.preprocessing.StandardScaler().fit(X['train'])
    X = {
        k: torch.tensor(preprocess.transform(v), device=device)
        for k, v in X.items()
    }
    y = {k: torch.tensor(v, device=device) for k, v in y.items()}

    return X, y, X_all, y_all, cat_cardinalities
