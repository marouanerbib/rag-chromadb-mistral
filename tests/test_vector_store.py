"""
Tests pour le gestionnaire de base de données vectorielle.
"""
import os
import shutil
import tempfile
import pytest
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store import VectorStoreManager

# Modèle d'embedding plus léger pour les tests
TEST_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def test_vector_store_initialization():
    """Teste l'initialisation du VectorStoreManager."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vsm = VectorStoreManager(
            persist_directory=temp_dir,
            collection_name="test_collection",
            embedding_model=TEST_EMBEDDING_MODEL,
            device="cpu"
        )
        
        assert vsm.persist_directory == str(Path(temp_dir).resolve())
        assert vsm.collection_name == "test_collection"
        assert vsm.embeddings is not None

def test_vector_store_persistence():
    """Teste la persistance du vector store."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Création initiale
        vsm = VectorStoreManager(
            persist_directory=temp_dir,
            collection_name="test_persistence",
            embedding_model=TEST_EMBEDDING_MODEL,
            device="cpu"
        )
        
        # Vérifier que le dossier de persistance est créé
        assert os.path.exists(temp_dir)
        
        # Tester l'initialisation du vector store
        vsm.init_vector_store()
        assert vsm.vector_store is not None
        
        # Vérifier que la collection existe
        collections = vsm.vector_store._client.list_collections()
        assert any(c.name == "test_persistence" for c in collections)

# Ajoutez d'autres tests pour les méthodes comme add_documents, similarity_search, etc.
