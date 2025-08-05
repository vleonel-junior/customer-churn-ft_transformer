"""
Script de test pour valider l'implémentation du modèle Sparse FTT+ pour la régression.

Ce script effectue des tests basiques pour s'assurer que le modèle fonctionne correctement.
"""

import torch
import numpy as np
import sys
import traceback

def test_model_creation():
    """Test la création du modèle."""
    print("Test 1: Création du modèle...")
    
    try:
        from sparse_ftt_plus_regression.model import InterpretableFTTPlusRegression
        
        model = InterpretableFTTPlusRegression.make_baseline(
            n_num_features=5,
            cat_cardinalities=[3, 4, 2],
            d_token=64,
            n_blocks=2,
            n_heads=4,
            attention_dropout=0.1,
            ffn_d_hidden=128,
            ffn_dropout=0.1,
            residual_dropout=0.0,
            d_out=1
        )
        
        print(f"  ✓ Modèle créé avec succès")
        print(f"  ✓ Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")
        return model
        
    except Exception as e:
        print(f"  ✗ Erreur lors de la création du modèle: {e}")
        traceback.print_exc()
        return None


def test_forward_pass(model):
    """Test le forward pass du modèle."""
    print("\nTest 2: Forward pass...")
    
    try:
        # Créer des données d'exemple (correspondant aux cardinalités du modèle)
        batch_size = 16
        X_num = torch.randn(batch_size, 5)
        X_cat = torch.randint(0, 3, (batch_size, 3))  # Cardinalités [3, 4, 2] -> max values [2, 3, 1]
        X_cat[:, 1] = torch.randint(0, 4, (batch_size,))  # Feature 1: cardinalité 4
        X_cat[:, 2] = torch.randint(0, 2, (batch_size,))  # Feature 2: cardinalité 2
        
        # Forward pass sans attention
        predictions = model(X_num, X_cat)
        
        print(f"  ✓ Forward pass réussi")
        print(f"  ✓ Shape des prédictions: {predictions.shape}")
        print(f"  ✓ Type des prédictions: {predictions.dtype}")
        
        # Forward pass avec attention
        predictions_with_attention, attention = model(X_num, X_cat, return_attention=True)
        
        print(f"  ✓ Forward pass avec attention réussi")
        print(f"  ✓ Shape de l'attention: {attention.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Erreur lors du forward pass: {e}")
        traceback.print_exc()
        return False


def test_loss_computation():
    """Test le calcul de la perte."""
    print("\nTest 3: Calcul de la perte...")
    
    try:
        from utils_regression import get_regression_loss_function
        
        # Test uniquement MSE (simple et efficace)
        loss_fn = get_regression_loss_function('mse')
        
        # Données de test
        predictions = torch.randn(10, 1)
        targets = torch.randn(10, 1)
        
        loss = loss_fn(predictions, targets)
        
        print(f"  ✓ Fonction de perte MSE: {loss.item():.6f}")
        print(f"  ✓ RMSE correspondant: {torch.sqrt(loss).item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Erreur lors du calcul de la perte: {e}")
        traceback.print_exc()
        return False


def test_metrics():
    """Test le calcul des métriques."""
    print("\nTest 4: Calcul des métriques...")
    
    try:
        from utils_regression import regression_performance_dict
        
        # Données de test
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        metrics = regression_performance_dict(y_true, y_pred)
        
        expected_keys = ['mse', 'rmse', 'mae', 'r2', 'mape']
        for key in expected_keys:
            if key not in metrics:
                print(f"  ✗ Métrique manquante: {key}")
                return False
            print(f"  ✓ {key}: {metrics[key]:.6f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Erreur lors du calcul des métriques: {e}")
        traceback.print_exc()
        return False


def test_feature_importance(model):
    """Test le calcul de l'importance des features."""
    print("\nTest 5: Importance des features...")
    
    try:
        # Données d'exemple (correspondant aux cardinalités du modèle)
        X_num = torch.randn(8, 5)
        X_cat = torch.randint(0, 3, (8, 3))  # Initialiser avec des valeurs valides
        X_cat[:, 1] = torch.randint(0, 4, (8,))  # Feature 1: cardinalité 4 (0-3)
        X_cat[:, 2] = torch.randint(0, 2, (8,))  # Feature 2: cardinalité 2 (0-1)
        feature_names = ['feat_num_0', 'feat_num_1', 'feat_num_2', 'feat_num_3', 'feat_num_4',
                        'feat_cat_0', 'feat_cat_1', 'feat_cat_2']
        
        # Test avec moyennage sur toutes les couches
        importance_all = model.get_cls_importance(X_num, X_cat, feature_names=feature_names, average_layers=True)
        
        print(f"  ✓ Importance (toutes couches): {len(importance_all)} features")
        
        # Test avec dernière couche seulement
        importance_last = model.get_cls_importance(X_num, X_cat, feature_names=feature_names, average_layers=False)
        
        print(f"  ✓ Importance (dernière couche): {len(importance_last)} features")
        
        # Vérifier que les scores sont des nombres
        for name, score in importance_all.items():
            if not isinstance(score, (int, float, np.floating)):
                print(f"  ✗ Score d'importance invalide pour {name}: {type(score)}")
                return False
        
        print(f"  ✓ Tous les scores d'importance sont valides")
        return True
        
    except Exception as e:
        print(f"  ✗ Erreur lors du calcul de l'importance: {e}")
        traceback.print_exc()
        return False


def test_optimization_groups(model):
    """Test les groupes de paramètres pour l'optimisation."""
    print("\nTest 6: Groupes de paramètres d'optimisation...")
    
    try:
        param_groups = model.optimization_param_groups()
        
        print(f"  ✓ Nombre de groupes de paramètres: {len(param_groups)}")
        
        total_params = 0
        for i, group in enumerate(param_groups):
            group_params = sum(p.numel() for p in group['params'])
            total_params += group_params
            weight_decay = group.get('weight_decay', 'default')
            print(f"  ✓ Groupe {i}: {group_params:,} paramètres (weight_decay: {weight_decay})")
        
        model_params = sum(p.numel() for p in model.parameters())
        if total_params != model_params:
            print(f"  ✗ Nombre de paramètres incohérent: {total_params} vs {model_params}")
            return False
        
        print(f"  ✓ Cohérence des paramètres vérifiée")
        return True
        
    except Exception as e:
        print(f"  ✗ Erreur lors du test des groupes de paramètres: {e}")
        traceback.print_exc()
        return False


def test_data_utils():
    """Test les utilitaires de données."""
    print("\nTest 7: Utilitaires de données...")
    
    try:
        from utils_regression import create_regression_dataset_example, normalize_targets, denormalize_predictions
        
        # Test création de données d'exemple
        X_num, X_cat, y, feature_names = create_regression_dataset_example()
        
        print(f"  ✓ Dataset d'exemple créé: {X_num.shape}, {X_cat.shape}, {y.shape}")
        print(f"  ✓ Feature names: {len(feature_names)} features")
        
        # Test normalisation
        y_norm, y_mean, y_std = normalize_targets(y)
        
        print(f"  ✓ Normalisation: mean={y_mean.item():.3f}, std={y_std.item():.3f}")
        
        # Test dénormalisation
        y_denorm = denormalize_predictions(y_norm, y_mean, y_std)
        
        # Vérifier que la dénormalisation récupère les valeurs originales
        diff = torch.abs(y - y_denorm).max()
        if diff > 1e-6:
            print(f"  ✗ Erreur de dénormalisation: diff max = {diff}")
            return False
        
        print(f"  ✓ Dénormalisation vérifiée (diff max: {diff:.2e})")
        return True
        
    except Exception as e:
        print(f"  ✗ Erreur lors du test des utilitaires: {e}")
        traceback.print_exc()
        return False


def main():
    """Fonction principale de test."""
    print("=== Tests du modèle Sparse FTT+ pour Régression ===\n")
    
    tests_passed = 0
    total_tests = 7
    
    # Test 1: Création du modèle
    model = test_model_creation()
    if model is not None:
        tests_passed += 1
    else:
        print("❌ Impossible de continuer sans modèle")
        sys.exit(1)
    
    # Test 2: Forward pass
    if test_forward_pass(model):
        tests_passed += 1
    
    # Test 3: Calcul de la perte
    if test_loss_computation():
        tests_passed += 1
    
    # Test 4: Métriques
    if test_metrics():
        tests_passed += 1
    
    # Test 5: Importance des features
    if test_feature_importance(model):
        tests_passed += 1
    
    # Test 6: Groupes de paramètres
    if test_optimization_groups(model):
        tests_passed += 1
    
    # Test 7: Utilitaires de données
    if test_data_utils():
        tests_passed += 1
    
    # Résumé
    print(f"\n=== Résumé des tests ===")
    print(f"Tests réussis: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 Tous les tests sont passés avec succès!")
        print("✅ Le modèle Sparse FTT+ pour régression est prêt à être utilisé.")
        return True
    else:
        print(f"❌ {total_tests - tests_passed} test(s) ont échoué.")
        print("⚠️  Veuillez corriger les erreurs avant d'utiliser le modèle.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)