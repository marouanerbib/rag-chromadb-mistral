from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rag_project",
    version="0.1.0",
    author="Votre Nom",
    author_email="votre.email@example.com",
    description="Un système RAG (Retrieval-Augmented Generation) pour l'indexation et la recherche de documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/votre-utilisateur/rag-project",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.2",
        "pypdf>=3.15.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.66.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
