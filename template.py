"""
template.py - Script pour créer automatiquement l'architecture du projet RAG
"""
import os
from pathlib import Path


def create_project_structure():
    """Crée la structure complète du projet RAG"""
    
    # Définir la structure du projet
    structure = {
        'src': {
            'files': [
                '__init__.py',
                'document_indexer.py',
                'vector_store.py',
                'document_retriever.py',
                'llm_qa_system.py',
                'evaluator.py',
                'chatbot.py'
            ],
            'utils': {
                'files': [
                    '__init__.py',
                    'config_loader.py',
                    'logger.py',
                    'metrics.py'
                ]
            }
        },
        'data': {
            'files': []
        }
    }
    
    # Fichiers à la racine
    root_files = [
        'config.yaml',
        'requirements.txt',
        'cli.py',
        'README.md',
        '.gitignore'
    ]
    
    # Créer les dossiers et fichiers
    def create_structure(base_path, struct):
        for name, content in struct.items():
            # Créer le dossier
            folder_path = base_path / name
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"Cree: {folder_path}")
            
            # Créer les fichiers dans ce dossier
            if 'files' in content:
                for file in content['files']:
                    file_path = folder_path / file
                    file_path.touch(exist_ok=True)
                    print(f"  Cree: {file_path}")
            
            # Récursion pour les sous-dossiers
            sub_folders = {k: v for k, v in content.items() if k != 'files'}
            if sub_folders:
                create_structure(folder_path, sub_folders)
    
    # Obtenir le chemin du projet
    project_root = Path.cwd()
    print(f"Création du projet dans: {project_root}\n")
    
    # Créer la structure
    create_structure(project_root, structure)
    
    # Créer les fichiers à la racine
    print(f"\nCreation des fichiers racine:")
    for file in root_files:
        file_path = project_root / file
        file_path.touch(exist_ok=True)
        print(f"Cree: {file_path}")
    
    print("\nStructure du projet creee avec succes!")
    print("\nStructure finale:")
    print_tree(project_root)


def print_tree(directory, prefix="", max_depth=3, current_depth=0):
    """Affiche l'arborescence du projet"""
    if current_depth >= max_depth:
        return
    
    try:
        contents = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        
        for i, path in enumerate(contents):
            # Ignorer certains dossiers
            if path.name in ['__pycache__', '.git', 'venv', 'env', '.idea']:
                continue
            
            is_last = i == len(contents) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{path.name}")
            
            if path.is_dir():
                extension = "    " if is_last else "│   "
                print_tree(path, prefix + extension, max_depth, current_depth + 1)
    except PermissionError:
        pass


if __name__ == "__main__":
    create_project_structure()
