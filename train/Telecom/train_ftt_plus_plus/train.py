"""
Script d'Entraînement FTT++ pour le Dataset Telecom

Ce script implémente l'entraînement complet du pipeline FTT++ en deux étapes :
1. Entraînement complet FTT+ → Analyse d'interprétabilité → Sélection des M features importantes
2. Entraînement Random → Modèle final avec attention sparse

Usage:
    python train.py --M 10 --k 5 --seed 0
    python train.py --M 15 --k 8 --stage1_epochs 100 --stage2_epochs 50
"""

import argparse
import torch

# Imports du projet avec la nouvelle architecture modulaire
from data.process_telecom_data import get_data
from ftt_plus_plus import FTTPlusPlusPipeline, FTTPlusPlusConfig, FeatureMapping
from ftt_plus_plus.training_stages import Stage1Trainer, Stage2Trainer


def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description='Entraînement FTT++ sur le dataset Telecom')
    
    # Paramètres FTT++
    parser.add_argument('--M', type=int, default=10, 
                       help='Nombre de features à sélectionner (défaut: 10)')
    parser.add_argument('--k', type=int, default=5,
                       help='Nombre d\'interactions feature-feature aléatoires (défaut: 5)')
    
    # Paramètres d'entraînement
    parser.add_argument('--seed', type=int, default=0,
                       help='Seed pour la reproductibilité (défaut: 0)')
    parser.add_argument('--stage1_epochs', type=int, default=50,
                       help='Époques pour l\'étape 1 (FTT+) (défaut: 50)')
    parser.add_argument('--stage2_epochs', type=int, default=50,
                       help='Époques pour l\'étape 2 (Random) (défaut: 50)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Taux d\'apprentissage (défaut: 1e-3)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Taille des batches (défaut: 64)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience pour early stopping (défaut: 10)')
    
    # Configuration du modèle
    parser.add_argument('--d_token', type=int, default=64,
                       help='Dimension des tokens (défaut: 64)')
    parser.add_argument('--n_blocks', type=int, default=2,
                       help='Nombre de blocs Transformer (défaut: 2)')
    parser.add_argument('--ffn_hidden', type=int, default=128,
                       help='Taille cachée du FFN (défaut: 128)')
    parser.add_argument('--embedding_type', type=str, default='LR',
                       help='Type d\'embedding numérique (défaut: LR)')
    
    # Paramètres de device
    parser.add_argument('--device', type=str, default=None,
                       help='Device à utiliser (cuda/cpu, défaut: auto-détection)')
    parser.add_argument('--force_cpu', action='store_true',
                       help='Forcer l\'utilisation du CPU même si GPU disponible')
    
    # Paramètres de sauvegarde
    parser.add_argument('--results_dir', type=str, default='results/results_telecom',
                       help='Répertoire de sauvegarde (défaut: results/results_telecom)')
    parser.add_argument('--save_intermediate', action='store_true',
                       help='Sauvegarder les résultats intermédiaires')
    
    return parser.parse_args()


def setup_device(args):
    """Configure le device à utiliser."""
    if args.force_cpu:
        device = 'cpu'
        print("Device forcé: CPU")
    elif args.device is not None:
        device = args.device
        print(f"Device spécifié: {device}")
    else:
        # Auto-détection
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU détecté: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("Aucun GPU détecté, utilisation du CPU")
        print(f"Device sélectionné: {device}")

    return device


def create_ftt_plus_plus_config(args) -> FTTPlusPlusConfig:
    """Crée la configuration FTT++ à partir des arguments."""
    
    # Configuration FTT+ (étape 1)
    ftt_plus_config = {
        'd_token': args.d_token,
        'n_blocks': args.n_blocks,
        'attention_dropout': 0.1,
        'ffn_d_hidden': args.ffn_hidden,
        'ffn_dropout': 0.1,
        'residual_dropout': 0.1,
        'd_out': 1
    }
    
    # Configuration Random (étape 2)
    random_model_config = {
        'd_token': args.d_token,
        'n_blocks': args.n_blocks,
        'attention_dropout': 0.1,
        'ffn_d_hidden': args.ffn_hidden,
        'ffn_dropout': 0.1,
        'residual_dropout': 0.1,
        'd_out': 1
    }
    
    return FTTPlusPlusConfig(
        ftt_plus_config=ftt_plus_config,
        M=args.M,
        k=args.k,
        random_model_config=random_model_config,
        attention_seed=args.seed,
        results_dir=args.results_dir,
        save_intermediate=args.save_intermediate
    )


def main():
    """Fonction principale."""
    args = parse_arguments()
    
    print("=== ENTRAÎNEMENT FTT++ MODULAIRE - DATASET TELECOM ===")
    
    # Configuration du device
    device = setup_device(args)
    
    # Charger les données
    print(f"Chargement des données (seed: {args.seed})")
    X, y, cat_cardinalities = get_data(args.seed)
    
    print(f"Features numériques: {X['train'][0].shape[1]}")
    print(f"Features catégorielles: {len(cat_cardinalities)}")
    print(f"Échantillons: train={len(y['train'])}, val={len(y['val'])}, test={len(y['test'])}")
    
    # Créer le mapping des features et la configuration
    feature_mapping = FeatureMapping.from_telecom_dataset()
    config = create_ftt_plus_plus_config(args)
    
    # Créer et exécuter le pipeline (gère tout : entraînement, analyse, sauvegarde)
    pipeline = FTTPlusPlusPipeline(config, feature_mapping=feature_mapping)
    
    pipeline.run_complete_pipeline(
        X=X,
        y=y,
        cat_cardinalities=cat_cardinalities,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
        embedding_type=args.embedding_type,
        device=device
    )


if __name__ == '__main__':
    main()