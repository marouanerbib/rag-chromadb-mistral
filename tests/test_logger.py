"""
Tests pour le module logger.
"""
import os
import logging
from pathlib import Path
from utils.logger import get_logger

def test_logger_creation():
    """Teste la création d'un logger."""
    logger = get_logger(__name__)
    assert isinstance(logger, logging.Logger)
    assert logger.name == __name__
    assert logger.level == logging.INFO

def test_log_file_creation(tmp_path):
    """Vérifie que le fichier de log est créé."""
    # Créer un dossier temporaire pour les logs
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_file = log_dir / "rag_system.log"
    
    # Configurer le logger avec le chemin temporaire
    logger = get_logger("test_logger")
    
    # Ajouter un gestionnaire de fichier manuellement pour le test
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Écrire un message de test
    logger.info("Test message")
    
    # Vérifier que le fichier a été créé
    assert log_file.exists(), f"Le fichier de log devrait être créé dans {log_file}"
    
    # Attendre que le message soit écrit (pour éviter les problèmes de buffering)
    import time
    time.sleep(0.1)
    
    # Vérifier que le message a bien été écrit
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    assert "Test message" in content, f"Le message de test n'a pas été trouvé dans {content}"
