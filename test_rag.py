#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de test pour le système RAG.
"""

import sys
from pathlib import Path

# Ajout du répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent))

# Import direct du module rag_system
from src.rag_system import RAGSystem

def main():
    """Fonction principale pour tester le système RAG."""
    print("Initialisation du système RAG...")
    
    try:
        # Initialisation du système RAG
        rag = RAGSystem()
        
        print("\nSystème RAG initialisé avec succès !")
        print("Tapez 'exit' pour quitter.")
        
        while True:
            # Demande de la question
            question = input("\nPosez votre question : ").strip()
            
            if question.lower() in ('exit', 'quit', 'q'):
                print("\nAu revoir !")
                break
                
            if not question:
                continue
                
            # Obtention de la réponse
            print("\nRecherche en cours...")
            result = rag.answer_question(question)
            
            # Affichage des résultats
            print("\n" + "=" * 80)
            print(f"QUESTION: {result['question']}")
            print("-" * 80)
            print(f"RÉPONSE: {result['answer']}")
            print("-" * 80)
            
            if result['source_documents']:
                print("\nSOURCES :")
                for i, doc in enumerate(result['source_documents'], 1):
                    source = doc.get('metadata', {}).get('source', 'Inconnu')
                    page = doc.get('metadata', {}).get('page', 'N/A')
                    print(f"{i}. {source} (Page {page})")
                    
                    # Afficher un extrait du contenu
                    content = doc.get('content', '')
                    if len(content) > 150:
                        content = content[:150] + "..."
                    print(f"   {content}\n")
            
            print("=" * 80)
            
    except Exception as e:
        print(f"\n❌ Une erreur est survenue : {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
