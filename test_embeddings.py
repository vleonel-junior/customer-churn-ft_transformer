"""
Test complet des embeddings numériques.
Ce script teste différents types d'embeddings numériques pour s'assurer de leur bon fonctionnement.
"""

import torch
import numpy as np
from num_embedding_factory import get_num_embedding

def test_embedding_type(embedding_type, n_features=8, d_embedding=32, batch_size=64):
    """
    Test un type d'embedding spécifique.
    
    Args:
        embedding_type: Type d'embedding à tester
        n_features: Nombre de features numériques
        d_embedding: Dimension de sortie
        batch_size: Taille du batch de test
    
    Returns:
        tuple: (success, error_message)
    """
    # Données d'entraînement avec échantillons suffisants
    train_size = 2000 if embedding_type.startswith(("Q", "T")) else 500
    X_train = np.random.randn(train_size, n_features)
    y_train = np.random.randint(0, 2, train_size)
    
    # Batch de test
    X_test = torch.randn(batch_size, n_features)
    
    # Configuration des paramètres selon le type
    params = {
        "embedding_type": embedding_type,
        "X_train": X_train,
        "d_embedding": d_embedding,
    }
    
    if embedding_type.startswith("T"):
        params["y_train"] = y_train
        params["n_bins"] = 8
    elif embedding_type.startswith("Q"):
        params["n_bins"] = 8
    elif embedding_type.startswith("P"):
        params["d_periodic_embedding"] = d_embedding
        params["sigma"] = 0.05
    
    try:
        # Création de l'embedder
        embedder = get_num_embedding(**params)
        
        # Forward pass
        with torch.no_grad():
            output = embedder(X_test)
        
        # Validation des résultats
        expected_shape = (batch_size, n_features, d_embedding)
        
        if output.shape != expected_shape:
            return False, f"Shape incorrecte: {output.shape} != {expected_shape}"
        
        if torch.isnan(output).any():
            return False, "Valeurs NaN détectées dans la sortie"
        
        if not torch.isfinite(output).all():
            return False, "Valeurs infinies détectées dans la sortie"
        
        return True, None
        
    except Exception as e:
        return False, str(e)

def run_comprehensive_test():
    """
    Exécute un test complet de tous les types d'embeddings.
    
    Returns:
        dict: Résultats détaillés des tests
    """
    # Configuration du test
    test_config = {
        "n_features": 10,
        "d_embedding": 64,
        "batch_size": 128
    }
    
    # Liste complète des types d'embeddings
    embedding_types = [
        # Simple embeddings
        "L", "LR", "LR-LR",
        
        # Quantile-based embeddings
        "Q", "Q-L", "Q-LR", "Q-LR-LR",
        
        # Tree-based embeddings
        "T", "T-L", "T-LR", "T-LR-LR",
        
        # Periodic embeddings
        "P", "P-L", "P-LR", "P-LR-LR"
    ]
    
    results = {}
    
    print("Test des embeddings numériques")
    print("-" * 60)
    print(f"Configuration: {test_config['n_features']} features, "
          f"dimension {test_config['d_embedding']}, "
          f"batch size {test_config['batch_size']}")
    print("-" * 60)
    
    for embedding_type in embedding_types:
        success, error = test_embedding_type(embedding_type, **test_config)
        results[embedding_type] = {
            "success": success,
            "error": error
        }
        
        # Affichage du résultat
        status = "PASS" if success else "FAIL"
        error_display = f" - {error}" if error else ""
        print(f"{embedding_type:<8} : {status}{error_display}")
    
    return results

def analyze_results(results):
    """
    Analyse et affiche un résumé des résultats de test.
    
    Args:
        results: Dictionnaire des résultats de test
    """
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r["success"])
    failed_tests = total_tests - passed_tests
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES TESTS")
    print("=" * 60)
    print(f"Total des tests     : {total_tests}")
    print(f"Tests réussis       : {passed_tests}")
    print(f"Tests échoués       : {failed_tests}")
    print(f"Taux de réussite    : {passed_tests/total_tests*100:.1f}%")
    
    if failed_tests > 0:
        print("\nTests échoués:")
        for emb_type, result in results.items():
            if not result["success"]:
                print(f"  {emb_type}: {result['error']}")
    
    print("\nRecommandations:")
    if passed_tests == total_tests:
        print("  Tous les embeddings sont fonctionnels.")
        print("  Embeddings recommandés pour la production:")
        print("    - P-LR  : Bon compromis performance/simplicité")
        print("    - Q-LR  : Haute performance sur la plupart des datasets")
        print("    - LR    : Baseline simple et efficace")
    else:
        working_embeddings = [emb for emb, result in results.items() if result["success"]]
        print(f"  Embeddings fonctionnels disponibles: {', '.join(working_embeddings)}")

def performance_benchmark():
    """
    Test de performance basique sur un échantillon plus important.
    """
    print("\n" + "=" * 60)
    print("BENCHMARK PERFORMANCE")
    print("=" * 60)
    
    # Configuration pour le benchmark
    X_train = np.random.randn(5000, 15)
    test_embeddings = ["LR", "P-LR", "Q-LR"]
    
    import time
    
    for emb_type in test_embeddings:
        try:
            # Préparation
            embedder = get_num_embedding(emb_type, X_train, d_embedding=128)
            x_test = torch.randn(512, 15)
            
            # Warm-up
            with torch.no_grad():
                _ = embedder(x_test)
            
            # Mesure du temps
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = embedder(x_test)
            elapsed_time = (time.time() - start_time) / 10
            
            print(f"{emb_type:<8} : {elapsed_time*1000:.2f} ms/batch (512 échantillons)")
            
        except Exception as e:
            print(f"{emb_type:<8} : Erreur - {str(e)}")

def main():
    """Fonction principale d'exécution des tests."""
    print("=" * 60)
    print("TEST PROFESSIONNEL DES EMBEDDINGS NUMÉRIQUES")
    print("=" * 60)
    
    # Test principal
    results = run_comprehensive_test()
    
    # Analyse des résultats
    analyze_results(results)
    
    # Benchmark de performance
    performance_benchmark()
    
    # Code de sortie
    success_rate = sum(1 for r in results.values() if r["success"]) / len(results)
    return success_rate >= 0.9  # Succès si au moins 90% des tests passent

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)