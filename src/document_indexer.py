"""
Module principal pour l'indexation de documents dans un système RAG.
"""
import logging
from typing import List, Optional, Union
from pathlib import Path

from langchain_core.documents import Document

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importations locales
from .document_loader import DocumentLoader
from .text_splitter import TextSplitter
from .vector_store import VectorStoreManager
from .utils.config_loader import ConfigLoader

class DocumentIndexer:
    """
    Classe principale pour l'indexation de documents dans un système RAG.
    
    Cette classe orchestre le chargement, le découpage et le stockage des documents.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise l'indexeur de documents avec la configuration.
        
        Args:
            config_path: Chemin vers le fichier de configuration YAML
        """
        # Initialisation du chargeur de configuration
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.config
        
        # Configuration des composants
        self._init_components()
        
        logger.info("DocumentIndexer initialisé avec succès")
    
    def _init_components(self):
        """Initialise les différents composants avec la configuration."""
        # Configuration du text splitter
        splitter_config = self.config.get('text_splitter', {})
        self.text_splitter = TextSplitter(
            chunk_size=splitter_config.get('chunk_size', 1000),
            chunk_overlap=splitter_config.get('chunk_overlap', 200),
            separators=splitter_config.get('separators', ["\n\n", "\n", " ", ""])
        )
        
        # Configuration du vector store et des embeddings
        vector_config = self.config.get('vector_store', {})
        embedding_config = self.config.get('embedding', {})
        
        self.vector_store = VectorStoreManager(
            persist_directory=vector_config.get('persist_directory', 'chroma_db'),
            collection_name=vector_config.get('collection_name', 'documents'),
            embedding_model=embedding_config.get('model_name', 'sentence-transformers/all-mpnet-base-v2'),
            device=embedding_config.get('device', 'cpu')
        )
    
    def load_documents(self, directory: Optional[Union[str, Path]] = None) -> List[Document]:
        """
        Charge les documents depuis un répertoire.
        
        Args:
            directory: Chemin vers le répertoire contenant les documents.
                      Si None, utilise le dossier par défaut de la configuration.
        
        Returns:
            Liste de documents chargés
        """
        if directory is None:
            directory = self.config.get("data", {}).get("raw_dir", "data/raw")
        
        logger.info(f"Chargement des documents depuis {directory}...")
        return DocumentLoader.load_from_directory(directory)
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Découpe les documents en chunks plus petits.
        
        Args:
            documents: Liste de documents à découper
            
        Returns:
            Liste de documents découpés
        """
        return self.text_splitter.split_documents(documents)
    
    def create_vector_store(self, documents: List[Document], force_recreate: bool = False):
        """
        Crée ou charge un vector store à partir des documents.
        
        Args:
            documents: Liste de documents à indexer
            force_recreate: Si True, force la recréation du vector store
            
        Returns:
            Instance du VectorStoreManager
        """
        if force_recreate:
            self.vector_store.delete_collection()
        
        if documents:
            self.vector_store.add_documents(documents)
            
        return self.vector_store

    def index_documents(self, directory: Optional[Union[str, Path]] = None, 
                       force_recreate: bool = False):
        """
        Charge, découpe et indexe les documents.
        
        Args:
            directory: Répertoire contenant les documents à indexer.
                     Si None, utilise le dossier par défaut de la configuration.
            force_recreate: Si True, force la recréation du vector store
            
        Returns:
            Instance du VectorStoreManager
        """
        # Chargement des documents
        documents = self.load_documents(directory)
        
        if not documents:
            logger.warning("Aucun document à indexer")
            return self.vector_store
        
        # Découpage des documents
        split_docs = self.split_documents(documents)
        
        # Création du vector store
        self.create_vector_store(split_docs, force_recreate)
        
        logger.info(f"Indexation terminée. {len(split_docs)} chunks indexés.")
        return self.vector_store

def main():
    """Fonction principale pour l'exécution en tant que script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Indexation de documents pour RAG")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Chemin vers le fichier de configuration"
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=None,
        help="Répertoire contenant les documents à indexer"
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Forcer la recréation du vector store"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialisation de l'indexeur
        indexer = DocumentIndexer(args.config)
        
        # Indexation des documents
        vector_store = indexer.index_documents(
            directory=args.directory,
            force_recreate=args.force_recreate
        )
        
        if vector_store and vector_store.vector_store:
            # Récupérer le nombre de documents indexés
            count = len(vector_store.vector_store.get()['ids'])
            print("\n✅ Indexation réussie !")
            print(f"📄 Documents indexés : {count}")
            print(f"📁 Dossier du vector store : {vector_store.persist_directory}")
            print("\nVous pouvez maintenant lancer l'interface de recherche avec :")
            print("  python recherche.py")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'indexation : {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()