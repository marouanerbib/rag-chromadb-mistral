"""
Tests pour le système RAG principal.
"""
import os
import pytest
import yaml
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY, mock_open
from typing import Dict, Any, List

# Importer le système RAG
from src.rag_system import RAGSystem

# Configuration de test
TEST_CONFIG = {
    'rag': {
        'llm': {
            'model_name': 'test-model',
            'model_path': 'models',
            'model_file': 'test-model.gguf',
            'temperature': 0.7,
            'max_new_tokens': 100,
            'max_length': 512,  # Ajout de la clé manquante
            'model_type': 'test',
            'gpu_layers': 0,
            'context_length': 2048
        },
        'retrieval': {
            'persist_directory': 'test_db',
            'embedding_model': 'test-embedding-model',
            'chunk_size': 1000,
            'chunk_overlap': 200
        },
        'prompt': {
            'system_prompt': 'Test system prompt',
            'template': 'Test template: {question}'
        }
    }
}

# Mock pour le modèle de langage
class MockLLM:
    def __init__(self, *args, **kwargs):
        self.config = kwargs
    
    def __call__(self, *args, **kwargs):
        return [{"generated_text": "Réponse générée"}]
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # Simuler le chargement du modèle
        return cls()

# Mock pour le vector store
class MockVectorStoreManager:
    def __init__(self, *args, **kwargs):
        self.persist_directory = kwargs.get('persist_directory')
    
    def add_documents(self, documents):
        return len(documents)
    
    def similarity_search(self, query, k=5):
        return ["Document pertinent"]

# Mock pour le processeur de documents
class MockDocumentProcessor:
    def __init__(self, *args, **kwargs):
        pass
    
    def load_documents(self, file_paths):
        return ["Document chargé"]
    
    def _process_documents(self, documents):
        return documents

# Mock pour les embeddings
class MockEmbeddings:
    def __init__(self, *args, **kwargs):
        self.model_name = kwargs.get('model_name')
    
    def embed_query(self, text):
        return [0.1] * 384

@patch('rag_system.AutoModelForCausalLM', MockLLM)
@patch('rag_system.VectorStoreManager', MockVectorStoreManager)
@patch('rag_system.DocumentProcessor', MockDocumentProcessor)
@patch('huggingface_hub.hf_hub_download', return_value="/fake/path/to/model.bin")
@patch('os.path.exists', return_value=True)
@patch('os.makedirs')
def test_rag_system_initialization(mock_makedirs, mock_exists, mock_download, MockDocumentProcessor, MockVectorStoreManager, MockLLM):
    """Teste l'initialisation du système RAG."""
    # Créer un fichier de configuration temporaire
    config_path = str(tmp_path / "test_config.yaml")
    
    # Simuler le contenu du fichier de configuration
    config_content = yaml.dump(TEST_CONFIG)
    
    # Utiliser mock_open pour simuler la lecture du fichier
    with patch('builtins.open', mock_open(read_data=config_content)) as mock_file:
        with patch('os.path.join', return_value="/fake/path/to/model.bin"):
            rag = RAGSystem(config_path)
            
            # Vérifier que le fichier a été ouvert avec le bon chemin
            mock_file.assert_called_once_with(config_path, 'r', encoding='utf-8')
    
    # Vérifications
    assert rag.config is not None
    assert hasattr(rag, 'vector_store')
    assert hasattr(rag, 'document_processor')
    assert hasattr(rag, 'model')

@patch('rag_system.AutoModelForCausalLM', MockLLM)
@patch('rag_system.VectorStoreManager', MockVectorStoreManager)
@patch('rag_system.DocumentProcessor', MockDocumentProcessor)
@patch('huggingface_hub.hf_hub_download', return_value="/fake/path/to/model.bin")
@patch('os.path.exists', return_value=True)
@patch('os.makedirs')
def test_generate_answer(mock_makedirs, mock_exists, mock_download, MockDocumentProcessor, MockVectorStoreManager, MockLLM, tmp_path):
    """Teste la génération de réponses."""
    # Créer un fichier de configuration temporaire
    config_path = str(tmp_path / "test_config.yaml")
    
    # Simuler le contenu du fichier de configuration
    config_content = yaml.dump(TEST_CONFIG)
    
    # Utiliser mock_open pour simuler la lecture du fichier
    with patch('builtins.open', mock_open(read_data=config_content)) as mock_file:
        with patch('os.path.join', return_value="/fake/path/to/model.bin"):
            rag = RAGSystem(config_path)
            
            # Vérifier que le fichier a été ouvert avec le bon chemin
            mock_file.assert_called_once_with(config_path, 'r', encoding='utf-8')
    
    # Simuler la recherche de documents
    rag.vector_store = MockVectorStoreManager()
    
    # Simuler le modèle
    rag.model = MockLLM()
    
    # Tester la génération de réponse
    response = rag.generate_answer("Question de test")
    
    # Vérifications
    assert "Réponse générée" in response

@patch('rag_system.AutoModelForCausalLM', MockLLM)
@patch('rag_system.VectorStoreManager', MockVectorStoreManager)
@patch('rag_system.DocumentProcessor', MockDocumentProcessor)
@patch('huggingface_hub.hf_hub_download', return_value="/fake/path/to/model.bin")
@patch('os.path.exists', return_value=True)
@patch('os.makedirs')
def test_add_documents(mock_makedirs, mock_exists, mock_download, MockDocumentProcessor, MockVectorStoreManager, MockLLM, tmp_path):
    """Teste l'ajout de documents."""
    # Créer un fichier de configuration temporaire
    config_path = str(tmp_path / "test_config.yaml")
    
    # Simuler le contenu du fichier de configuration
    config_content = yaml.dump(TEST_CONFIG)
    
    # Utiliser mock_open pour simuler la lecture du fichier
    with patch('builtins.open', mock_open(read_data=config_content)) as mock_file:
        with patch('os.path.join', return_value="/fake/path/to/model.bin"):
            rag = RAGSystem(config_path)
            
            # Vérifier que le fichier a été ouvert avec le bon chemin
            mock_file.assert_called_once_with(config_path, 'r', encoding='utf-8')
    
    # Simuler le vector store
    rag.vector_store = MockVectorStoreManager()
    
    # Simuler le processeur de documents
    rag.document_processor = MockDocumentProcessor()
    
    # Tester l'ajout de documents
    test_docs = ["doc1.pdf", "doc2.pdf"]
    result = rag.add_documents(test_docs)
    
    # Vérifications
    assert result == len(test_docs)
