"""
Module pour le chargement des documents depuis différents formats.
"""
from typing import List, Union, Optional
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader,
    UnstructuredFileLoader
)
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Classe pour charger des documents depuis différents formats."""
    
    @staticmethod
    def load_single_document(file_path: Union[str, Path]) -> List[Document]:
        """Charge un seul document selon son extension."""
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_extension == '.txt':
                loader = TextLoader(str(file_path))
            elif file_extension == '.md':
                loader = UnstructuredMarkdownLoader(str(file_path))
            else:
                # Pour les autres formats, on essaie avec UnstructuredFileLoader
                loader = UnstructuredFileLoader(str(file_path))
                
            return loader.load()
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {file_path}: {str(e)}")
            return []
    
    @classmethod
    def load_from_directory(
        cls, 
        directory: Union[str, Path], 
        glob_pattern: str = "**/*.*"
    ) -> List[Document]:
        """Charge tous les documents d'un répertoire."""
        directory = Path(directory)
        all_docs = []
        
        for file_path in directory.glob(glob_pattern):
            if file_path.is_file():
                docs = cls.load_single_document(file_path)
                for doc in docs:
                    doc.metadata["source"] = str(file_path)
                all_docs.extend(docs)
                
        logger.info(f"{len(all_docs)} documents chargés depuis {directory}")
        return all_docs
