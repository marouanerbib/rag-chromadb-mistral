"""
Module de configuration des logs pour l'application.
"""
import logging
import os
import sys
from pathlib import Path
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue

# File pour stocker les logs
LOG_FILE = "logs/rag_system.log"
LOG_LEVEL = logging.INFO

# Créer le dossier de logs s'il n'existe pas
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configuration de base du logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_logger(name: str) -> logging.Logger:
    """
    Crée et configure un logger avec un gestionnaire de fichier et de console.
    
    Args:
        name: Nom du logger (généralement __name__)
        
    Returns:
        Un logger configuré
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # Éviter la propagation vers le logger racine pour éviter les doublons
    logger.propagate = False
    
    # Si le logger a déjà des gestionnaires, le retourner tel quel
    if logger.handlers:
        return logger
    
    # Créer le formateur
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Gestionnaire pour la console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Gestionnaire pour le fichier
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Ajouter les gestionnaires au logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger