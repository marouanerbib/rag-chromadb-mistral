#!/usr/bin/env python3
"""
Interface en ligne de commande pour le système RAG.
"""
import argparse
import sys
import subprocess
import yaml
from pathlib import Path
from src.document_indexer import DocumentIndexer
from src.rag_system import RAGSystem
from src.utils.logger import get_logger

# Configuration du logger
logger = get_logger(__name__)

def parse_arguments():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Système RAG pour l'indexation et la recherche de documents"
    )
    
    # Arguments principaux
    subparsers = parser.add_subparsers(dest="command", help="Commandes disponibles")
    
    # Commande index
    index_parser = subparsers.add_parser("index", help="Indexer des documents")
    index_parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Répertoire contenant les documents à indexer"
    )
    index_parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Forcer la recréation complète de l'index"
    )
    index_parser.add_argument(
        "--config",
        type=str,
        default="config/config_rag.yaml",
        help="Chemin vers le fichier de configuration"
    )
    
    # Commande ask
    ask_parser = subparsers.add_parser("ask", help="Poser une question au système RAG")
    ask_parser.add_argument(
        "question",
        type=str,
        help="Question à poser au système"
    )
    ask_parser.add_argument(
        "--config",
        type=str,
        default="config/config_rag.yaml",
        help="Chemin vers le fichier de configuration"
    )
    
    # Commande test
    test_parser = subparsers.add_parser("test", help="Exécuter les tests")
    test_parser.add_argument(
        "--coverage",
        action="store_true",
        help="Générer un rapport de couverture"
    )
    test_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Afficher plus de détails"
    )
    test_parser.add_argument(
        "test_path",
        nargs="?",
        default="tests/",
        help="Chemin vers les tests à exécuter (par défaut: tests/)"
    )
    
    # Commande evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Évaluer les performances du système RAG")
    eval_parser.add_argument(
        "--test-data",
        type=str,
        default="tests/fixtures/evaluation/test_set.json",
        help="Chemin vers le fichier de données de test (par défaut: tests/fixtures/evaluation/test_set.json)"
    )
    eval_parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Fichier de sortie pour les résultats d'évaluation (par défaut: evaluation_results.json)"
    )
    eval_parser.add_argument(
        "--config",
        type=str,
        default="config/config_rag.yaml",
        help="Chemin vers le fichier de configuration du système RAG"
    )
    
    return parser.parse_args()

def run_tests(args):
    """Exécute les tests avec les options spécifiées."""
    cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        cmd.append("-v")
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
    
    cmd.append(args.test_path)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error("Certains tests ont échoué")
        sys.exit(1)

def index_documents(args):
    """Indexe les documents du répertoire spécifié."""
    try:
        logger.info(f"Indexation des documents dans {args.input_dir}")
        indexer = DocumentIndexer(config_path=args.config)
        indexer.index_directory(args.input_dir, force_recreate=args.force_recreate)
        logger.info("Indexation terminée avec succès")
    except Exception as e:
        logger.error(f"Erreur lors de l'indexation: {str(e)}")
        sys.exit(1)

def ask_question(args):
    """Pose une question au système RAG."""
    try:
        logger.info(f"Chargement du système RAG avec la configuration: {args.config}")
        rag = RAGSystem(config_path=args.config)
        logger.info("Système RAG chargé avec succès")
        
        logger.info(f"Traitement de la question: {args.question}")
        response = rag.answer_question(args.question)
        
        print("\n" + "="*80)
        print(f"QUESTION: {args.question}")
        print("-"*80)
        print(f"RÉPONSE: {response['answer']}")
        print("="*80 + "\n")
        
        if response.get('sources'):
            print("SOURCES :")
            for i, source in enumerate(response['sources'], 1):
                print(f"{i}. {source.get('source', 'N/A')} (Page {source.get('page', 'N/A')})")
                print("   " + source.get('content', '')[:200] + "...\n")
                
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la question: {str(e)}")
        sys.exit(1)

def evaluate_system(args):
    """Évalue les performances du système RAG."""
    try:
        from src.evaluation.evaluator import RAGEvaluator
        import json
        from pathlib import Path
        
        logger.info(f"Chargement des données de test depuis {args.test_data}")
        
        # Vérifier que le fichier de test existe
        if not Path(args.test_data).exists():
            logger.error(f"Fichier de test introuvable : {args.test_data}")
            sys.exit(1)
        
        # Initialiser l'évaluateur
        evaluator = RAGEvaluator()
        
        # Charger les données de test
        with open(args.test_data, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        logger.info(f"Évaluation en cours sur {len(test_data)} exemples...")
        
        # Exécuter l'évaluation
        results = evaluator.evaluate_end_to_end(test_data)
        
        # Enregistrer les résultats
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Afficher un résumé
        print("\n=== Résumé de l'évaluation ===")
        print(f"Nombre d'échantillons évalués: {len(test_data)}")
        
        print("\nMétriques de récupération (moyennes) :")
        print(f"- Précision: {results['retrieval_metrics']['avg_precision']:.3f}")
        print(f"- Rappel: {results['retrieval_metrics']['avg_recall']:.3f}")
        print(f"- F1-score: {results['retrieval_metrics']['avg_f1_score']:.3f}")
        
        print("\nMétriques de génération (moyennes) :")
        print(f"- Précision: {results['generation_metrics']['avg_precision']:.3f}")
        print(f"- Rappel: {results['generation_metrics']['avg_recall']:.3f}")
        print(f"- F1-score: {results['generation_metrics']['avg_f1_score']:.3f}")
        print(f"- Exact match: {results['generation_metrics']['avg_exact_match']:.3f}")
        
        print(f"\nRésultats détaillés enregistrés dans : {output_path.absolute()}")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation : {str(e)}", exc_info=True)
        sys.exit(1)

def main():
    """Fonction principale du CLI."""
    args = parse_arguments()
    
    if not args.command:
        print("Veuillez spécifier une commande. Utilisez --help pour l'aide.")
        sys.exit(1)
    
    try:
        if args.command == "index":
            index_documents(args)
        elif args.command == "ask":
            ask_question(args)
        elif args.command == "test":
            run_tests(args)
        elif args.command == "evaluate":
            evaluate_system(args)
        else:
            print(f"Commande inconnue: {args.command}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\nOpération annulée par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Erreur: {str(e)}", exc_info=True)
        sys.exit(1)
    
    else:
        print("Commande non reconnue. Utilisez 'index' ou 'search'.")

if __name__ == "__main__":
    main()