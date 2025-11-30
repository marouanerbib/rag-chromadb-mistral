"""
Module pour charger et gérer la configuration de l'application.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Classe pour charger et gérer la configuration de l'application.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise le chargeur de configuration.
        
        Args:
            config_path: Chemin vers le fichier de configuration.
        """
        try:
            # Définition des chemins par défaut
            self.base_dir = Path(__file__).parent.parent.parent
            
            # Vérifier si le chemin de configuration est fourni
            if config_path is None:
                config_path = self.base_dir / "config" / "config.yaml"
            else:
                config_path = Path(config_path)
                if not config_path.is_absolute():
                    config_path = self.base_dir / config_path
            
            logger.info(f"Chargement de la configuration depuis : {config_path}")
            
            # Chargement de la configuration
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
            
            # Initialisation des chemins
            self._init_paths()
            
            logger.info("Configuration chargée avec succès")
            
        except FileNotFoundError as e:
            logger.error(f"Fichier de configuration non trouvé : {e}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Erreur de syntaxe dans le fichier de configuration : {e}")
            raise
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration : {e}")
            raise
    
    def _init_paths(self) -> None:
        """Initialise les chemins de l'application."""
        # Chemins de base
        data_config = self.config.get('data', {})
        
        # Chemins relatifs par rapport au répertoire de base
        self.data_dir = Path(data_config.get('raw_dir', 'data'))
        self.raw_data_dir = self.base_dir / data_config.get('raw_dir', 'data/raw')
        self.processed_data_dir = self.base_dir / data_config.get('processed_dir', 'data/processed')
        
        # Création des répertoires s'ils n'existent pas
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def get_pdf_files(self) -> List[Path]:
        """
        Retourne la liste des fichiers PDF dans le dossier d'entrée.
        
        Returns:
            Liste des chemins vers les fichiers PDF
        """
        return list(self.raw_data_dir.glob("**/*.pdf"))
    
    def get_vector_store_path(self) -> Path:
        """
        Retourne le chemin du stockage vectoriel.
        
        Returns:
            Chemin du dossier de stockage vectoriel
        """
        return self.base_dir / self.config['vector_store']['persist_directory']
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration d'embedding.
        
        Returns:
            Dictionnaire de configuration pour l'embedding
        """
        return self.config.get('embedding', {})
    
    def get_text_splitter_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration du découpage de texte.
        
        Returns:
            Dictionnaire de configuration pour le découpage de texte
        """
        return self.config.get('text_splitter', {})