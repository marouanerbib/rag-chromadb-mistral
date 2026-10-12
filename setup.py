from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rag_project",
    version="0.1.0",
    author="Marouane Rbib",
    description="Un système RAG (Retrieval-Augmented Generation) pour l'indexation et la recherche de documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marouanerbib/rag-chromadb-mistral",
    packages=find_packages(),
    install_requires=[
        "langchain-core>=1.0",
        "langchain-community>=0.4",
        "langchain-text-splitters>=1.0",
        "chromadb>=1.0",
        "sentence-transformers>=5.0",
        "ctransformers>=0.2.27",
        "pypdf>=6.0",
        "pyyaml>=6.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
