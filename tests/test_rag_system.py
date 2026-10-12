"""
Tests pour le système RAG principal.
Le modèle LLM et le vector store sont simulés : les tests ne téléchargent rien.
"""
from unittest.mock import patch, MagicMock

import pytest
import yaml
from langchain_core.documents import Document

from src.rag_system import RAGSystem

TEST_CONFIG = {
    'rag': {
        'llm': {
            'model_name': 'test-model',
            'model_file': 'test-model.gguf',
            'temperature': 0.7,
            'max_new_tokens': 100,
            'max_length': 512,
        },
        'retrieval': {
            'persist_directory': 'test_db',
            'embedding_model': 'test-embedding-model',
            'top_k': 2,
            'min_score': 0.3,
            'chunk_size': 1000,
            'chunk_overlap': 200,
        },
        'prompt': {
            'system_prompt': 'Test system prompt',
            'template': '[INST] {system_prompt}\nContexte: {context}\nQuestion: {question} [/INST]',
        },
    }
}


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config_rag.yaml"
    path.write_text(yaml.dump(TEST_CONFIG), encoding="utf-8")
    return str(path)


@pytest.fixture
def rag(config_path):
    with patch("src.rag_system.VectorStoreManager") as vsm, \
         patch("src.rag_system.DocumentProcessor"), \
         patch.object(RAGSystem, "_initialize_llm_components"):
        system = RAGSystem(config_path)
    system.vector_store = vsm.return_value
    system.model = MagicMock(return_value="Réponse générée")
    return system


def test_rag_system_initialization(rag):
    assert rag.config["llm"]["repo_id"] == "test-model"
    assert rag.config["retrieval"]["top_k"] == 2


def test_missing_section_raises(tmp_path):
    bad = {"rag": {"llm": TEST_CONFIG["rag"]["llm"], "retrieval": {}}}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.dump(bad), encoding="utf-8")
    with pytest.raises(KeyError):
        RAGSystem(str(path))


def test_answer_question_returns_answer_and_sources(rag):
    docs = [Document(page_content="Le délai est de 14 jours.", metadata={"source": "doc.pdf", "page": 3})]
    rag.vector_store.similarity_search.return_value = docs

    result = rag.answer_question("Quel est le délai ?")

    assert result["answer"] == "Réponse générée"
    assert len(result["source_documents"]) == 1
    assert result["source_documents"][0]["metadata"]["source"] == "doc.pdf"
    rag.vector_store.similarity_search.assert_called_once_with(
        query="Quel est le délai ?", k=2, score_threshold=0.3
    )
    prompt = rag.model.call_args.args[0]
    assert "Le délai est de 14 jours." in prompt
    assert "Quel est le délai ?" in prompt


def test_answer_question_without_relevant_documents(rag):
    rag.vector_store.similarity_search.return_value = []

    result = rag.answer_question("Question hors corpus")

    assert result["source_documents"] == []
    assert result["metadata"]["retrieved_docs"] == 0
    rag.model.assert_not_called()


def test_empty_question_is_rejected(rag):
    result = rag.answer_question("   ")
    assert "error" in result["metadata"]
