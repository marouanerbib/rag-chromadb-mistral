"""
Configuration des tests et fixtures partagées.
"""
import os
import shutil
import tempfile
import pytest
from pathlib import Path

# Ajout du répertoire src au PYTHONPATH
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

@pytest.fixture(scope="session")
def test_data_dir():
    """Crée un répertoire temporaire pour les données de test."""
    test_dir = Path(tempfile.mkdtemp(prefix="rag_test_"))
    yield test_dir
    # Nettoyage après les tests
    if test_dir.exists():
        shutil.rmtree(test_dir)

@pytest.fixture(scope="session")
def sample_pdf_path(test_data_dir):
    """Retourne le chemin vers un fichier PDF de test."""
    # Créer un PDF de test simple
    pdf_path = test_data_dir / "test_document.pdf"
    # TODO: Ajouter un vrai PDF de test ou utiliser un mock
    return str(pdf_path)
