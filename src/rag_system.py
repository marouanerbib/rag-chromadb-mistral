"""
Module implémentant un système RAG (Retrieval-Augmented Generation) qui combine
des capacités de recherche vectorielle avec un modèle de langage pour fournir
des réponses basées sur des documents.
"""

import os
import json
import yaml
import torch
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from ctransformers import AutoModelForCausalLM, AutoConfig

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from .document_indexer import DocumentIndexer
from .vector_store import VectorStoreManager
from .document_processor import DocumentProcessor
from .utils.logger import get_logger

# Initialisation du logger
logger = get_logger(__name__)


class RAGSystem:
    """
    Système RAG qui combine recherche vectorielle et génération de texte.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise le système RAG avec la configuration fournie.
        
        Args:
            config_path: Chemin vers le fichier de configuration YAML
        """
        logger.info("Initialisation du système RAG...")
        
        # Chargement de la configuration
        if config_path is None:
            config_path = os.path.join("config", "config_rag.yaml")
        
        logger.info(f"Chargement de la configuration depuis : {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                if not config_data or 'rag' not in config_data:
                    error_msg = "Configuration 'rag' non trouvée dans le fichier de configuration"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                self.config = config_data['rag']
                
                logger.debug("Configuration chargée : %s", json.dumps(self.config, indent=2, ensure_ascii=False))
                
                # Vérification des clés requises
                required_keys = ['llm', 'retrieval', 'prompt']
                for key in required_keys:
                    if key not in self.config:
                        error_msg = f"Section '{key}' manquante dans la configuration"
                        logger.error(error_msg)
                        raise KeyError(error_msg)
                
                # Vérification des clés requises pour llm
                llm_required = ['model_name', 'temperature', 'max_length']
                for key in llm_required:
                    if key not in self.config['llm']:
                        error_msg = f"Clé '{key}' manquante dans la section 'llm'"
                        logger.error(error_msg)
                        raise KeyError(error_msg)
                
                # Utilisation de model_name comme repo_id si repo_id n'est pas spécifié
                if 'repo_id' not in self.config['llm']:
                    self.config['llm']['repo_id'] = self.config['llm']['model_name']
                    logger.debug("Utilisation de model_name comme repo_id: %s", self.config['llm']['repo_id'])
        
        except FileNotFoundError as e:
            error_msg = f"Fichier de configuration non trouvé : {config_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg) from e
            
        except yaml.YAMLError as e:
            error_msg = f"Erreur de syntaxe dans le fichier de configuration : {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        
        # Initialisation du vector store
        logger.info("Initialisation du vector store...")
        self.vector_store = VectorStoreManager(
            persist_directory=self.config['retrieval'].get('persist_directory', 'chroma_db'),
            collection_name=self.config['retrieval'].get('collection_name', 'documents'),
            embedding_model=self.config['retrieval'].get('embedding_model', 'sentence-transformers/all-mpnet-base-v2'),
            device=self.config['retrieval'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Initialisation du processeur de documents
        self.document_processor = DocumentProcessor(
            chunk_size=self.config['retrieval'].get('chunk_size', 1000),
            chunk_overlap=self.config['retrieval'].get('chunk_overlap', 200)
        )
        
        # Initialisation du modèle LLM
        logger.info("Initialisation du modèle LLM...")
        logger.debug("Configuration du modèle: %s", self.config['llm'])
        
        # Initialisation des composants du modèle
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        try:
            self._initialize_llm_components()
            logger.info("Modèle LLM initialisé avec succès: %s", self.config['llm']['repo_id'])
            
        except Exception as e:
            error_msg = f"Erreur lors de l'initialisation du modèle HuggingFace : {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
        
        # Initialisation du template de prompt
        logger.debug("Chargement du template de prompt...")
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=self.config['prompt'].get('template', 
                """Répondez à la question en vous basant sur le contexte fourni. Si vous ne connaissez pas la réponse, dites-le clairement.
                
                Contexte:
                {context}
                
                Question: {question}
                
                Réponse:""")
        )
        
        logger.info("Système RAG initialisé avec succès")
    
    def _initialize_llm_components(self):
        """Initialise les composants du modèle de langage avec ctransformers."""
        model_name = self.config['llm']['model_name']
        model_path = os.path.abspath(self.config['llm'].get('model_path', 'models'))
        model_file = self.config['llm'].get('model_file')
        model_type = self.config['llm'].get('model_type', 'mistral')
        local_model = self.config['llm'].get('local_model', False)
        
        # Configuration du modèle
        config = {
            'model_type': model_type,
            'context_length': int(self.config['llm'].get('context_length', 2048)),
            'gpu_layers': int(self.config['llm'].get('gpu_layers', 0)),  # 0 pour CPU uniquement
            'threads': max(1, os.cpu_count() // 2),  # Utiliser la moitié des cœurs disponibles
            'batch_size': 8,  # Taille de lot pour l'inférence
            'max_new_tokens': int(self.config['llm'].get('max_new_tokens', 512)),
            'temperature': float(self.config['llm'].get('temperature', 0.7)),
            'top_p': float(self.config['llm'].get('top_p', 0.9)),
            'top_k': int(self.config['llm'].get('top_k', 40)),
            'repetition_penalty': float(self.config['llm'].get('repetition_penalty', 1.1)),
            'last_n_tokens': 64,
            'seed': 42,
            'reset': True
        }
        
        # Créer le répertoire du modèle s'il n'existe pas
        os.makedirs(model_path, exist_ok=True)
        
        try:
            model_file_path = os.path.join(model_path, model_file) if model_file else None
            
            # Si le modèle est déjà téléchargé localement
            if model_file and os.path.exists(model_file_path):
                logger.info(f"Chargement du modèle local depuis {model_file_path}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    model_file=model_file,
                    **{k: v for k, v in config.items()}
                )
            # Si on veut forcer l'utilisation d'un modèle local mais qu'il n'existe pas
            elif local_model:
                error_msg = f"Modèle local {model_file} non trouvé dans {model_path}. " \
                          f"Veuillez télécharger le modèle manuellement ou désactiver 'local_model' dans la configuration."
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            # Téléchargement depuis Hugging Face Hub
            else:
                logger.warning(f"Tentative de téléchargement du modèle {model_name} depuis Hugging Face Hub...")
                logger.warning("Cette opération peut prendre du temps (environ 3 Go à télécharger).")
                
                try:
                    from huggingface_hub import hf_hub_download
                    
                    # Télécharger le fichier du modèle
                    model_file_path = hf_hub_download(
                        repo_id=model_name,
                        filename=model_file,
                        cache_dir=model_path,
                        resume_download=True
                    )
                    
                    logger.info(f"Modèle téléchargé avec succès dans : {model_file_path}")
                    
                    # Charger le modèle depuis le fichier téléchargé
                    self.model = AutoModelForCausalLM.from_pretrained(
                        os.path.dirname(model_file_path),
                        model_file=os.path.basename(model_file_path),
                        **{k: v for k, v in config.items()}
                    )
                except Exception as e:
                    error_msg = f"Échec du téléchargement du modèle : {str(e)}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
            
            logger.info("Modèle chargé avec succès")
            
        except Exception as e:
            error_msg = f"Erreur lors du chargement du modèle: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def _call_huggingface_api(self, prompt: str, **generation_kwargs) -> str:
        """
        Appelle le modèle pour générer une réponse à partir d'un prompt.
        
        Args:
            prompt: Le prompt à envoyer au modèle
            **generation_kwargs: Arguments supplémentaires pour la génération
            
        Returns:
            La réponse générée par le modèle
            
        Raises:
            RuntimeError: Si une erreur survient lors de la génération de la réponse
        """
        logger = logging.getLogger(__name__)
        
        # Fusionner les paramètres de génération avec les valeurs par défaut
        generation_params = {
            'max_new_tokens': int(self.config['llm'].get('max_new_tokens', 512)),
            'temperature': float(self.config['llm'].get('temperature', 0.7)),
            'top_p': float(self.config['llm'].get('top_p', 0.9)),
            'top_k': int(self.config['llm'].get('top_k', 40)),
            'repetition_penalty': float(self.config['llm'].get('repetition_penalty', 1.1)),
            'seed': 42,
            'reset': True,
            'stream': False  # Désactiver le streaming pour une réponse complète
        }
        
        # Mettre à jour avec les paramètres fournis
        generation_params.update(generation_kwargs)
        
        logger.debug(f"Génération avec les paramètres: {generation_params}")
        logger.debug(f"Prompt: {prompt[:200]}..." if len(prompt) > 200 else f"Prompt: {prompt}")
        
        try:
            # Vérifier si le modèle est initialisé
            if not hasattr(self, 'model') or self.model is None:
                raise RuntimeError("Le modèle n'a pas été correctement initialisé")
            
            # Appeler le modèle
            response = self.model(prompt, **generation_params)
            
            # Si la réponse est une liste, prendre le premier élément
            if isinstance(response, list) and len(response) > 0:
                response = response[0]
            
            # Si la réponse est un dictionnaire, essayer d'extraire le texte
            if isinstance(response, dict):
                if 'text' in response:
                    response = response['text']
                elif 'generated_text' in response:
                    response = response['generated_text']
            
            # Si la réponse est un objet avec un attribut 'generations'
            if hasattr(response, 'generations') and response.generations:
                response = response.generations[0][0].text
            
            # Vérifier si la réponse est vide ou invalide
            if not response or not isinstance(response, str) or len(response.strip()) == 0:
                error_msg = "La réponse du modèle est vide ou invalide"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Nettoyer la réponse (supprimer les espaces superflus, etc.)
            response = response.strip()
            
            # Suppression du prompt de la réponse si nécessaire
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            # Nettoyage supplémentaire pour les modèles instruct
            if '[/INST]' in response:
                response = response.split('[/INST]', 1)[-1].strip()
            
            # Supprimer les balises de début et de fin
            response = response.replace('<s>', '').replace('</s>', '').strip()
            
            # Journaliser la réponse (tronquée si trop longue)
            logger.debug(f"Réponse générée: {response[:200]}..." if len(response) > 200 else f"Réponse générée: {response}")
            
            return response
            
        except Exception as e:
            error_msg = f"Erreur lors de la génération de la réponse : {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
            
    def _format_context(self, documents: List[Document]) -> str:
        """
        Formate les documents en une seule chaîne de contexte avec leurs métadonnées.
        
        Args:
            documents: Liste de documents à formater
            
        Returns:
            Chaîne formatée contenant le contexte
        """
        if not documents:
            return "Aucun document pertinent trouvé pour répondre à la question."
        
        formatted_docs = []
        
        for i, doc in enumerate(documents, 1):
            # Extraire les métadonnées pertinentes
            metadata = doc.metadata
            source = metadata.get('source', 'Source inconnue')
            page = metadata.get('page', 'N/A')
            score = metadata.get('score', 'N/A')
            
            # Formater les informations du document
            doc_info = f"Document {i} (Source: {source}, Page: {page}, Score: {score}):\n"
            
            # Ajouter le contenu du document
            content = doc.page_content.strip()
            
            # Construire la chaîne finale pour ce document
            formatted_docs.append(f"{doc_info}{content}")
        
        # Joindre tous les documents avec des séparateurs clairs
        return "\n\n" + "\n" + "-"*80 + "\n".join(formatted_docs) + "\n" + "-"*80 + "\n"

    def answer_question(
        self, 
        question: str, 
        top_k: Optional[int] = None,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Répond à une question en utilisant le système RAG.
        
        Args:
            question: Question à poser
            top_k: Nombre de documents à récupérer (optionnel, utilise la valeur par défaut si None)
            **generation_kwargs: Arguments supplémentaires pour la génération du modèle
            
        Returns:
            Dictionnaire contenant:
            - answer: La réponse générée
            - question: La question posée
            - source_documents: Liste des documents sources
            - context_used: Contexte utilisé pour générer la réponse
            - metadata: Métadonnées supplémentaires
        """
        logger.info(f"Traitement de la question : {question}")
        
        # Initialiser la liste des documents sources
        source_docs = []
        
        try:
            # Validation de la question
            if not question or not question.strip():
                raise ValueError("La question ne peut pas être vide")
            
            # Récupération des documents pertinents
            top_k = top_k or self.config['retrieval'].get('top_k', 2)
            min_score = self.config['retrieval'].get('min_score', 0.3)
            
            logger.info(f"Recherche des {top_k} documents les plus pertinents (score min: {min_score})...")
            
            # Recherche des documents pertinents
            retrieved_docs = self.vector_store.similarity_search(
                query=question,
                k=top_k,
                score_threshold=min_score
            )
            
            logger.info(f"{len(retrieved_docs)} documents retenus après filtrage par score")
            
            if not retrieved_docs:
                logger.warning("Aucun document pertinent trouvé pour la question")
                return {
                    "answer": "Désolé, je n'ai pas trouvé d'informations pertinentes pour répondre à votre question.",
                    "question": question,
                    "source_documents": [],
                    "context_used": "Aucun document pertinent trouvé.",
                    "metadata": {
                        "retrieved_docs": 0,
                        "warning": "Aucun document pertinent trouvé"
                    }
                }
            
            # Formatage du contexte
            context = self._format_context(retrieved_docs)
            
            # Vérification de la longueur du contexte
            max_context_length = self.config['retrieval'].get('max_context_length', 2000)
            if len(context) > max_context_length:
                logger.warning(f"Le contexte dépasse la taille maximale ({len(context)} > {max_context_length}), il sera tronqué")
                context = context[:max_context_length]
            
            # Génération de la réponse
            logger.info("Génération de la réponse par le modèle LLM...")
            
            # Préparation du prompt avec le format Mistral Instruct
            system_prompt = self.config['prompt'].get('system_prompt', '')
            prompt = self.config['prompt']['template'].format(
                system_prompt=system_prompt,
                context=context,
                question=question
            )
            
            try:
                # Appel direct au modèle avec ctransformers
                answer = self._call_huggingface_api(prompt, **generation_kwargs)
                
                # Vérification que la réponse n'est pas vide
                if not answer or not isinstance(answer, str) or not answer.strip():
                    raise ValueError("La réponse du modèle est vide ou invalide")
                
                # Nettoyage de la réponse
                answer = answer.strip()
                
                # Suppression des balises de fin de réponse si présentes
                if '[/INST]' in answer:
                    answer = answer.split('[/INST]')[0].strip()
                
                # Formatage des documents sources
                source_docs = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    } for doc in retrieved_docs
                ]
                
                # Retour de la réponse
                return {
                    "answer": answer,
                    "question": question,
                    "source_documents": source_docs,
                    "context_used": context,
                    "metadata": {
                        "retrieved_docs": len(retrieved_docs),
                        "context_length": len(context)
                    }
                }
                
            except Exception as e:
                logger.error(f"Erreur lors de la génération de la réponse : {str(e)}", exc_info=True)
                raise RuntimeError(f"Erreur lors de la génération de la réponse : {str(e)}") from e
                
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la question : {str(e)}", exc_info=True)
            return {
                "answer": f"Une erreur est survenue lors du traitement de votre question : {str(e)}",
                "question": question,
                "source_documents": [],
                "context_used": "",
                "metadata": {
                    "error": str(e),
                    "retrieved_docs": 0
                }
            }
