# Système RAG : ChromaDB + Mistral 7B en local

Pipeline RAG (Retrieval-Augmented Generation) qui indexe des documents PDF ou TXT, retrouve les passages pertinents par recherche sémantique et génère une réponse avec un LLM exécuté en local sur CPU, en citant ses sources.

## Fonctionnement

```
documents (PDF/TXT)
   │  chargement : PyPDFLoader / TextLoader
   ▼
découpage en chunks (RecursiveCharacterTextSplitter, 1000 caractères, recouvrement 200)
   │
   ▼
embeddings sentence-transformers/all-mpnet-base-v2  ──►  ChromaDB (persistant)
                                                              │
question ──► embedding ──► top-k par similarité, filtre score ≥ 0.3
                                                              │
                                                              ▼
                              prompt [INST] contexte + question ──► Mistral 7B Instruct (GGUF, ctransformers)
                                                              │
                                                              ▼
                                                  réponse + documents sources
```

## Stack

Python, LangChain (core, community, text-splitters), ChromaDB, sentence-transformers, ctransformers (Mistral 7B Instruct GGUF), pytest.

## Installation

```bash
git clone https://github.com/marouanerbib/rag-chromadb-mistral.git
cd rag-chromadb-mistral
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Modèle : télécharger `mistral-7b-instruct-v0.1.Q2_K.gguf` depuis [TheBloke/Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF) et le placer dans `models/`. Le chemin et le fichier se règlent dans `config/config_rag.yaml`. Avec `local_model: false`, le modèle est téléchargé automatiquement au premier lancement (environ 3 Go).

Les fichiers Markdown passent par `UnstructuredMarkdownLoader` et nécessitent en plus `pip install unstructured`.

## Utilisation

```bash
# 1. Indexer un dossier de documents
python cli.py index --input-dir data/raw

# 2. Poser une question (réponse + sources avec page)
python cli.py ask "Quel est le délai de rétractation pour un contrat à distance ?"

# Recherche sémantique interactive, sans LLM
python recherche.py

# Évaluation sur un jeu annoté
python cli.py evaluate --test-data tests/fixtures/evaluation/test_set.json

# Tests
python cli.py test
```

## Configuration

| Fichier | Rôle |
|---|---|
| `config/config.yaml` | indexation : dossier source, modèle d'embedding, collection Chroma, taille des chunks |
| `config/config_rag.yaml` | question-réponse : paramètres du LLM, top-k, score minimal, prompt système |
| `config/eval_config.yaml` | paramètres d'évaluation |

Le prompt système par défaut cible un assistant juridique (droit marocain) ; il se change dans `config_rag.yaml`.

## Évaluation

`src/evaluation/evaluator.py` calcule :
- pour la recherche : précision, rappel et F1 entre documents récupérés et documents pertinents ;
- pour la génération : précision, rappel et F1 sur les tokens par rapport à une réponse de référence, et exact match.

Il évalue un fichier JSON qui contient déjà les documents récupérés et les réponses générées. Le jeu fourni dans `tests/fixtures` est un exemple de deux questions, pas un benchmark.

## Structure

```
cli.py                    point d'entrée : index, ask, evaluate, test
recherche.py              recherche sémantique interactive
config/                   fichiers YAML
src/
  document_loader.py      chargement des fichiers d'un dossier
  document_processor.py   chargement + découpage d'une liste de fichiers
  text_splitter.py        découpage en chunks
  vector_store.py         embeddings et ChromaDB (ajout, recherche avec score)
  document_indexer.py     orchestration de l'indexation
  search.py               recherche et mise en forme des résultats
  rag_system.py           recherche + prompt + génération
  evaluation/evaluator.py métriques
  utils/                  configuration et logs
tests/                    pytest (le LLM et la base vectorielle sont simulés)
```

## Limites

- Le modèle quantifié Q2_K tient sur CPU mais dégrade la qualité des réponses ; un quantifié Q4 ou Q5 est un meilleur compromis si la mémoire le permet.
- Le contexte envoyé au LLM est tronqué à 2000 caractères.
- `tests/test_vector_store.py` télécharge un modèle d'embedding et nécessite un accès à Hugging Face.
