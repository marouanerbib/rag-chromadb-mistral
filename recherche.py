#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script d'interface de recherche pour le projet RAG
Permet d'effectuer des recherches dans les documents indexés
"""

import os
import sys
from pathlib import Path

# Ajout du répertoire parent au path pour les imports
sys.path.append(str(Path(__file__).parent))

from src.vector_store import VectorStoreManager
from src.document_indexer import DocumentIndexer
from src.search import search_documents, format_search_results

def clear_screen():
    """Efface l'écran de la console"""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_header():
    """Affiche l'en-tête de l'application"""
    clear_screen()
    print("""
    ******************************************
    *     SYSTÈME DE RECHERCHE DE DOCUMENTS  *
    *     Projet RAG - Indexation avancée    *
    ******************************************
    """)

def display_help():
    """Affiche l'aide"""
    print("\nCommandes disponibles :")
    print("  /help     - Affiche cette aide")
    print("  /clear    - Efface l'écran")
    print("  /reindex - Réindexe les documents")
    print("  /quit     - Quitte le programme")
    print("\nPour effectuer une recherche, tapez simplement votre requête.")

def reindex_documents():
    """Réindexe tous les documents"""
    print("\nDébut de la réindexation des documents...")
    try:
        indexer = DocumentIndexer("config/config.yaml")
        indexer.index_documents("data/raw", force_recreate=True)
        print("Réindexation terminée avec succès !")
    except Exception as e:
        print(f"Erreur lors de la réindexation : {str(e)}")

def search_loop():
    """Boucle principale de recherche"""
    try:
        # Vérification de l'initialisation du vector store
        VectorStoreManager()
    except Exception as e:
        print(f"Erreur lors de l'initialisation : {str(e)}")
        print("Assurez-vous d'avoir d'abord indexé des documents avec 'python cli.py index --input-dir data/raw'.")
        return
    
    print("\nSystème de recherche prêt. Tapez /help pour l'aide.")
    
    while True:
        try:
            # Demande de la requête
            query = input("\nEntrez votre recherche : ").strip()
            
            # Commandes spéciales
            if not query:
                continue
                
            if query.lower() == '/quit':
                print("Au revoir !")
                break
                
            if query.lower() == '/help':
                display_help()
                continue
                
            if query.lower() == '/clear':
                clear_screen()
                display_header()
                continue
                
            if query.lower() == '/reindex':
                reindex_documents()
                continue
            
            # Recherche
            print(f"\n🔍 Recherche : {query}")
            print("=" * 50)
            
            # Recherche des documents pertinents
            results = search_documents(query, k=5, min_score=0.3)
            
            # Affichage des résultats formatés
            print(format_search_results(results))
            
        except KeyboardInterrupt:
            print("\nUtilisez la commande /quit pour quitter.")
        except Exception as e:
            print(f"\n❌ Erreur lors de la recherche : {str(e)}")

def main():
    """Fonction principale"""
    display_header()
    
    # Vérifier si le dossier de données existe
    if not os.path.exists("data/raw") or not any(os.scandir("data/raw")):
        print("⚠️  Aucun document trouvé dans le dossier 'data/raw'.")
        print("Veuillez y ajouter des fichiers PDF, TXT ou MD avant de continuer.")
        return
    
    # Lancer la boucle de recherche
    search_loop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Erreur inattendue : {str(e)}")
        sys.exit(1)
