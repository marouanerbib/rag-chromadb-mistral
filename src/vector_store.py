"""
Module pour la gestion du stockage vectoriel des documents.
"""
from typing import List, Optional, Dict, Any
from pathlib import Path
import shutil
import logging

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Gère le stockage et la récupération des embeddings de documents."""
    
    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "documents",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cpu"
    ):
        """
        Initialise le gestionnaire de base vectorielle.
        
        Args:
            persist_directory: Dossier de persistance des données
            collection_name: Nom de la collection
            embedding_model: Nom du modèle d'embedding
            device: Appareil à utiliser (cpu/cuda)
        """
        self.persist_directory = str(Path(persist_directory).resolve())
        self.collection_name = collection_name
        
        # Désactiver les messages de warning de tqdm
        from tqdm import tqdm
        tqdm.disable = True
        
        # Configuration des timeouts pour les requêtes HTTP
        import os
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5 minutes de timeout
        os.environ['HF_HUB_ETAG_TIMEOUT'] = '60'      # 1 minute pour la vérification ETag
        os.environ['HF_HUB_DOWNLOAD_RETRY_DELAY'] = '10'  # 10 secondes entre les tentatives
        
        # Configuration minimale pour éviter les conflits
        model_kwargs = {
            'device': device,
            'trust_remote_code': True
        }
        
        # Initialisation du modèle d'embedding avec configuration minimale
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs=model_kwargs
        )
        
        # Désactiver explicitement la barre de progression dans le modèle sous-jacent
        if hasattr(self.embeddings.client, 'model') and hasattr(self.embeddings.client.model, 'eval'):
            self.embeddings.client.model.eval()
        
        # Désactiver les gradients pour l'inférence
        import torch
        torch.set_grad_enabled(False)
        
        self.vector_store = None
    
    def init_vector_store(self):
        """Initialise ou charge la base vectorielle."""
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
        return self.vector_store
    
    def add_documents(self, documents: List[Document]):
        """
        Ajoute des documents à la base vectorielle.
        
        Args:
            documents: Liste de documents à ajouter
        """
        if not documents:
            logger.warning("Aucun document à ajouter")
            return
            
        if self.vector_store is None:
            self.init_vector_store()
            
        logger.info(f"Ajout de {len(documents)} documents à la base vectorielle...")
        self.vector_store.add_documents(documents)
        self.vector_store.persist()
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        score_threshold: float = 0.0,
        **kwargs
    ) -> List[Document]:
        """
        Effectue une recherche de similarité et retourne les documents les plus pertinents.
        
        Args:
            query: La requête de recherche
            k: Nombre de documents à retourner
            score_threshold: Seuil de score minimum pour inclure un document
            **kwargs: Arguments supplémentaires pour la recherche
            
        Returns:
            Liste des documents les plus pertinents
        """
        if self.vector_store is None:
            self.init_vector_store()
            
        # Si un seuil est spécifié, utiliser similarity_search_with_relevance_scores
        if score_threshold > 0:
            docs_and_scores = self.vector_store.similarity_search_with_relevance_scores(
                query, k=k, **kwargs
            )
            
            # Filtrer par seuil de score
            filtered_docs = [
                doc for doc, score in docs_and_scores 
                if score >= score_threshold
            ]
            
            # Ajouter le score aux métadonnées
            for i, (doc, score) in enumerate(docs_and_scores):
                if i < len(filtered_docs):
                    filtered_docs[i].metadata['score'] = float(score)
            
            return filtered_docs
        else:
            # Recherche simple sans seuil
            return self.vector_store.similarity_search(query, k=k, **kwargs)
    
    def search(
        self, 
        query: str, 
        k: int = 5, 
        **search_kwargs
    ) -> List[tuple[Document, float]]:
        """
        Effectue une recherche sémantique avec score de similarité.
        
        Args:
            query: Requête de recherche
            k: Nombre de résultats à retourner
            **search_kwargs: Arguments supplémentaires de recherche
            
        Returns:
            Liste de tuples (document, score)
        """
        if self.vector_store is None:
            self.init_vector_store()
            
        return self.vector_store.similarity_search_with_score(
            query, 
            k=k, 
            **search_kwargs
        )
    
    def delete_collection(self):
        """Supprime la collection actuelle."""
        if Path(self.persist_directory).exists():
            shutil.rmtree(self.persist_directory)
            logger.info(f"Collection supprimée : {self.persist_directory}")
        
        self.vector_store = None