"""
Module pour le découpage des documents en chunks.
"""
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class TextSplitter:
    """Classe pour le découpage de documents en chunks."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initialise le text splitter.
        
        Args:
            chunk_size: Taille maximale d'un chunk
            chunk_overlap: Chevauchement entre les chunks
            separators: Liste des séparateurs de texte
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        
        self._init_splitter()
    
    def _init_splitter(self):
        """Initialise le splitter avec la configuration actuelle."""
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Découpe les documents en chunks.
        
        Args:
            documents: Liste de documents à découper
            
        Returns:
            Liste de documents découpés
        """
        if not documents:
            return []
            
        logger.info(f"Découpage de {len(documents)} documents...")
        return self.splitter.split_documents(documents)
    
    def update_parameters(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None
    ):
        """Met à jour les paramètres du splitter."""
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap
        if separators is not None:
            self.separators = separators
            
        self._init_splitter()
