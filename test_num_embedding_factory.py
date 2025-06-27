#!/usr/bin/env python3
"""
Test complet pour valider get_num_embedding
"""
import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Import de la fonction get_num_embedding
# Option 1: Si elle est dans un fichier séparé (remplacez 'num_embeddings' par le nom de votre fichier)
# from num_embeddings import get_num_embedding

# Option 2: Si elle est dans le même fichier, copiez-collez la fonction ici
# Ou importez depuis le bon module selon votre structure

# Pour ce test, ajoutez l'import correct ici :
try:
    from num_embedding_factory import get_num_embedding
except ImportError:
    print("❌ ERREUR: Impossible d'importer get_num_embedding")
    print("📝 Solutions:")
    print("1. Si votre fonction est dans 'num_embeddings.py': from num_embeddings import get_num_embedding")
    print("2. Si elle est dans un autre fichier: adaptez l'import")
    print("3. Ou copiez-collez la fonction get_num_embedding directement dans ce fichier")
    exit(1)

def test_synthetic_data():
    """Test avec des données synthétiques (batch_size=10, n_features=5)"""
    print("=" * 60)
    print("TEST AVEC DONNÉES SYNTHÉTIQUES")
    print("=" * 60)
    
    # Créer des données synthétiques
    np.random.seed(42)
    batch_size, n_features = 10, 5
    
    X_train = np.random.randn(100, n_features)  # Training set plus large
    y_train = np.random.randint(0, 2, 100)      # Labels binaires
    X_test = np.random.randn(batch_size, n_features)  # Batch de test
    
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    
    print(f"Shape X_train: {X_train.shape}")
    print(f"Shape X_test: {X_test.shape}")
    print(f"Shape y_train: {y_train.shape}")
    print()
    
    # Tester tous les types d'embedding
    embedding_types = [
        'L', 'LR', 'LR-LR',
        'Q', 'Q-L', 'Q-LR', 'Q-LR-LR',
        'T', 'T-L', 'T-LR', 'T-LR-LR',
        'P', 'P-L', 'P-LR', 'P-LR-LR'
    ]
    
    d_embedding = 8
    d_periodic_embedding = 6
    
    results = {}
    
    for emb_type in embedding_types:
        try:
            print(f"Testing {emb_type}...")
            
            # Créer l'embedding
            if emb_type.startswith('T'):
                embedder = get_num_embedding(
                    emb_type, X_train, d_embedding, 
                    y_train=y_train, n_bins=4
                )
            elif emb_type.startswith('P'):
                embedder = get_num_embedding(
                    emb_type, X_train, d_embedding,
                    d_periodic_embedding=d_periodic_embedding
                )
            else:
                embedder = get_num_embedding(
                    emb_type, X_train, d_embedding, n_bins=4
                )
            
            # Test forward pass
            with torch.no_grad():
                output = embedder(X_test_tensor)
                expected_shape = (batch_size, n_features, d_embedding)
                
                # Vérifier la forme
                shape_ok = output.shape == expected_shape
                
                # Vérifier que ce ne sont pas des NaN
                no_nan = not torch.isnan(output).any()
                
                # Vérifier la variance (pas tous zéros)
                has_variance = output.var() > 1e-6
                
                results[emb_type] = {
                    'shape': output.shape,
                    'shape_ok': shape_ok,
                    'no_nan': no_nan,
                    'has_variance': has_variance,
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item()
                }
                
                status = "✅" if (shape_ok and no_nan and has_variance) else "❌"
                print(f"  {emb_type:8} : {output.shape} {status}")
                
        except Exception as e:
            print(f"  {emb_type:8} : ERREUR - {str(e)}")
            results[emb_type] = {'error': str(e)}
    
    return results

def test_telecom_style_data():
    """Test avec des données style telecom"""
    print("\n" + "=" * 60)
    print("TEST AVEC DONNÉES STYLE TELECOM")
    print("=" * 60)
    
    # Générer des données qui ressemblent à un dataset telecom
    np.random.seed(42)
    n_samples_train = 200
    batch_size = 10
    
    # Features typiques telecom
    feature_names = [
        'monthly_charges', 'total_charges', 'tenure_months', 
        'data_usage_gb', 'call_minutes'
    ]
    n_features = len(feature_names)
    
    # Créer des données réalistes
    X_train = np.column_stack([
        np.random.uniform(20, 120, n_samples_train),    # monthly_charges
        np.random.uniform(100, 8000, n_samples_train),  # total_charges  
        np.random.randint(1, 73, n_samples_train),      # tenure_months
        np.random.exponential(5, n_samples_train),      # data_usage_gb
        np.random.uniform(0, 1000, n_samples_train)     # call_minutes
    ])
    
    # Labels: churn (0/1)
    y_train = np.random.binomial(1, 0.3, n_samples_train)
    
    # Batch de test
    X_test = np.column_stack([
        np.random.uniform(20, 120, batch_size),
        np.random.uniform(100, 8000, batch_size),
        np.random.randint(1, 73, batch_size),
        np.random.exponential(5, batch_size),
        np.random.uniform(0, 1000, batch_size)
    ])
    
    # Normalisation (optionnelle mais recommandée)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Conversion en tenseurs
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    
    print(f"Features: {feature_names}")
    print(f"X_train shape: {X_train_scaled.shape}")
    print(f"X_test shape: {X_test_scaled.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Churn rate: {y_train.mean():.2%}")
    print()
    
    # Statistiques des features
    print("Statistiques des features (après normalisation):")
    for i, name in enumerate(feature_names):
        print(f"  {name:15}: mean={X_train_scaled[:, i].mean():.3f}, "
              f"std={X_train_scaled[:, i].std():.3f}")
    print()
    
    # Test avec quelques embeddings représentatifs
    test_embeddings = ['LR', 'Q-LR', 'T-LR', 'P-LR']
    d_embedding = 16
    
    print("Test des embeddings sur données telecom:")
    for emb_type in test_embeddings:
        try:
            if emb_type.startswith('T'):
                embedder = get_num_embedding(
                    emb_type, X_train_scaled, d_embedding,
                    y_train=y_train, n_bins=5
                )
            elif emb_type.startswith('P'):
                embedder = get_num_embedding(
                    emb_type, X_train_scaled, d_embedding,
                    d_periodic_embedding=10
                )
            else:
                embedder = get_num_embedding(
                    emb_type, X_train_scaled, d_embedding, n_bins=5
                )
            
            with torch.no_grad():
                output = embedder(X_test_tensor)
                expected_shape = (batch_size, n_features, d_embedding)
                
                shape_ok = output.shape == expected_shape
                no_nan = not torch.isnan(output).any()
                has_variance = output.var() > 1e-6
                
                status = "✅" if (shape_ok and no_nan and has_variance) else "❌"
                print(f"  {emb_type:8} : {output.shape} | "
                      f"mean={output.mean():.3f} | "
                      f"std={output.std():.3f} {status}")
                
                # Afficher quelques valeurs pour le premier sample
                if emb_type == 'LR':  # Juste pour un exemple
                    print(f"    Premier sample, première feature (shape={output[0, 0].shape}):")
                    print(f"    {output[0, 0, :5].numpy()}")  # 5 premières dims
                
        except Exception as e:
            print(f"  {emb_type:8} : ERREUR - {str(e)}")

def test_gradient_flow():
    """Test que les gradients passent bien"""
    print("\n" + "=" * 60)
    print("TEST DU GRADIENT FLOW")
    print("=" * 60)
    
    batch_size, n_features, d_embedding = 5, 3, 4
    
    X_train = np.random.randn(50, n_features)
    X_test = torch.randn(batch_size, n_features, requires_grad=True)
    
    embedder = get_num_embedding('LR', X_train, d_embedding)
    
    # Forward pass
    output = embedder(X_test)
    
    # Créer une loss fictive
    target = torch.randn_like(output)
    loss = torch.nn.functional.mse_loss(output, target)
    
    # Backward pass
    loss.backward()
    
    # Vérifier que les gradients existent
    has_grad = X_test.grad is not None and not torch.allclose(X_test.grad, torch.zeros_like(X_test.grad))
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Input gradient exists and non-zero: {'✅' if has_grad else '❌'}")
    print(f"Input gradient norm: {X_test.grad.norm().item():.6f}")

def main():
    """Lance tous les tests"""
    print("VALIDATION COMPLÈTE DE get_num_embedding")
    print("=" * 60)
    
    # Test 1: Données synthétiques
    synthetic_results = test_synthetic_data()
    
    # Test 2: Données telecom
    test_telecom_style_data()
    
    # Test 3: Gradient flow
    test_gradient_flow()
    
    # Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES TESTS SYNTHÉTIQUES")
    print("=" * 60)
    
    success_count = 0
    total_count = 0
    
    for emb_type, result in synthetic_results.items():
        if 'error' not in result:
            total_count += 1
            if result.get('shape_ok') and result.get('no_nan') and result.get('has_variance'):
                success_count += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"{emb_type:8} : {status} | "
                  f"mean={result['mean']:6.3f} | "
                  f"std={result['std']:6.3f}")
        else:
            total_count += 1
            print(f"{emb_type:8} : ❌ | ERROR: {result['error']}")
    
    print(f"\nSuccès: {success_count}/{total_count} embeddings")
    print("\n🎉 Tests terminés ! Vous pouvez supprimer ce fichier.")

if __name__ == "__main__":
    main()