"""
Tests pour le processeur de documents.
"""
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from typing import List, Dict, Any

# Importer le processeur de documents
from src.document_processor import DocumentProcessor

# Mock pour les documents
class MockDocument:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

# Mocks pour les chargeurs
def mock_pdf_loader(file_path, **kwargs):
    class MockPDFLoader:
        def __init__(self, file_path, **kwargs):
            self.file_path = file_path
            
        def load(self):
            return [MockDocument("Contenu PDF de test", {"source": self.file_path, "page": 1})]
    return MockPDFLoader(file_path, **kwargs)

def mock_text_loader(file_path, **kwargs):
    class MockTextLoader:
        def __init__(self, file_path, **kwargs):
            self.file_path = file_path
            
        def load(self):
            return [MockDocument("Contenu texte de test", {"source": self.file_path})]
    return MockTextLoader(file_path, **kwargs)

@patch('src.document_processor.PyPDFLoader', side_effect=mock_pdf_loader)
@patch('src.document_processor.TextLoader', side_effect=mock_text_loader)
def test_document_processor_initialization(mock_text, mock_pdf):
    """Teste l'initialisation du processeur de documents."""
    processor = DocumentProcessor()
    assert processor is not None
    assert hasattr(processor, 'load_documents')
    assert hasattr(processor, 'process_documents')

@patch('src.document_processor.PyPDFLoader', side_effect=mock_pdf_loader)
@patch('src.document_processor.TextLoader', side_effect=mock_text_loader)
def test_load_documents(mock_text, mock_pdf):
    """Teste le chargement de documents."""
    # Tester le chargement de PDF
    processor = DocumentProcessor()
    result = processor.load_documents(["test.pdf"])
    assert len(result) == 1
    assert "Contenu PDF" in result[0].page_content
    mock_pdf.assert_called_once_with("test.pdf")
    
    # Tester le chargement de texte
    result = processor.load_documents(["test.txt"])
    assert len(result) == 1
    assert "Contenu texte" in result[0].page_content
    mock_text.assert_called_once_with("test.txt", encoding='utf-8')

@patch('src.document_processor.PyPDFLoader', side_effect=mock_pdf_loader)
@patch('src.document_processor.TextLoader', side_effect=mock_text_loader)
def test_process_documents(mock_text, mock_pdf):
    """Teste le traitement des documents."""
    # Tester le traitement d'un document
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    
    # Créer un fichier de test
    test_file = "test_doc.txt"
    with open(test_file, 'w') as f:
        f.write(" ".join(["mot"] * 1000))  # Contenu assez long pour être divisé
    
    try:
        # Tester le traitement
        result = processor.process_documents([test_file])
        
        # Vérifications
        # Le test vérifie qu'on a au moins un résultat
        assert len(result) >= 1
        assert all(hasattr(doc, 'page_content') for doc in result)
        
        # Vérifier que le contenu est correct
        assert any("Contenu texte" in doc.page_content for doc in result)
        
    finally:
        # Nettoyer
        if os.path.exists(test_file):
            os.remove(test_file)

def test_process_empty_documents():
    """Teste le traitement d'une liste vide de documents."""
    processor = DocumentProcessor()
    result = processor.process_documents([])
    assert len(result) == 0
