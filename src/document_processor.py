from typing import List, Dict, Any
from pathlib import Path
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Classe pour charger et prétraiter les documents avant leur indexation.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialise le processeur de documents.
        
        Args:
            chunk_size: Taille maximale des chunks
            chunk_overlap: Chevauchement entre les chunks
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Charge les documents depuis les chemins de fichiers fournis.
        
        Args:
            file_paths: Liste des chemins vers les fichiers à charger
            
        Returns:
            Liste des documents chargés
        """
        documents = []
        for file_path in file_paths:
            try:
                file_path = str(file_path)
                logger.info(f"Chargement du fichier: {file_path}")
                
                if file_path.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                    logger.info(f"{len(docs)} pages chargées depuis {file_path}")
                    
                elif file_path.lower().endswith(('.txt', '.md')):
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    documents.extend(docs)
                    logger.info(f"Document texte chargé: {file_path}")
                    
            except Exception as e:
                logger.error(f"Erreur lors du chargement de {file_path}: {str(e)}")
                continue
                
        return documents
    
    def process_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Charge et découpe les documents en chunks.
        
        Args:
            file_paths: Liste des chemins vers les fichiers à traiter
            
        Returns:
            Liste des chunks de documents
        """
        # Charger les documents
        documents = self.load_documents(file_paths)
        
        if not documents:
            logger.warning("Aucun document chargé à traiter")
            return []
        
        # Découper les documents en chunks
        chunks = self.splitter.split_documents(documents)
        logger.info(f"{len(chunks)} chunks créés à partir de {len(documents)} documents")
        
        return chunks
