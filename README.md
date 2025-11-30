# Système RAG (Retrieval-Augmented Generation)

Un système de génération de réponses basé sur la récupération d'informations, conçu pour fournir des réponses précises à partir d'un corpus de documents.

## 🚀 Fonctionnalités

- **Indexation de documents** : Prétraitement et stockage efficace des documents
- **Recherche contextuelle** : Récupération précise des documents pertinents
- **Génération de réponses** : Production de réponses naturelles basées sur le contexte
- **Évaluation intégrée** : Mesure des performances avec des métriques standardisées

## � Configuration du Modèle

### Téléchargement du Modèle Mistral

Ce projet utilise le modèle Mistral 7B. Suivez ces étapes pour le configurer :

1. **Télécharger le modèle** :
   - [Télécharger Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
   - Cliquez sur "Files and versions" puis téléchargez tous les fichiers

2. **Créer le dossier des modèles** :
   ```bash
   mkdir -p models/mistral-7b
   ```

3. **Placer les fichiers** :
   - Extrayez les fichiers téléchargés dans `models/mistral-7b/`
   - La structure doit ressembler à :
     ```
     models/
     └── mistral-7b/
         ├── config.json
         ├── model.safetensors
         ├── tokenizer.json
         └── ...
     ```

4. **Vérifier la configuration** :
   Assurez-vous que le fichier `config/config_rag.yaml` contient :
   ```yaml
   llm:
     model_path: "./models/mistral-7b"
   ```

## �📦 Prérequis

- Python 3.8+
- pip (gestionnaire de paquets Python)
- Un environnement virtuel Python (recommandé)

## 🛠 Installation

1. **Cloner le dépôt**
   ```bash
   git clone [https://github.com/Akiraaymane/RAG-Project.git]
   cd RAG-Project
   ```

2. **Créer et activer un environnement virtuel**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

## 🏗 Structure du Projet

```
.
├── config/                 # Fichiers de configuration
│   ├── config_rag.yaml     # Configuration principale
│   └── eval_config.yaml   # Configuration de l'évaluation
├── data/                   # Dossier des données
│   ├── raw/               # Documents bruts (PDF, TXT, etc.)
│   └── evaluation/        # Données pour l'évaluation
├── src/                   # Code source
│   ├── evaluation/        # Module d'évaluation
│   ├── models/            # Modèles et logique métier
│   └── utils/             # Utilitaires et helpers
└── tests/                 # Tests unitaires et d'intégration
```

## 🚀 Utilisation

### Indexation des documents
```bash
python cli.py index --input-dir data/raw/
```

### Poser une question
```bash
python cli.py ask "Votre question ici"
```

### Évaluer le système
```bash
python cli.py evaluate --test-data tests/fixtures/evaluation/test_set.json
```

## 📊 Métriques d'Évaluation

Le système fournit plusieurs métriques pour évaluer les performances :

- **Récupération** :
  - Précision : Proportion de documents pertinents parmi ceux récupérés
  - Rappel : Proportion de documents pertinents effectivement récupérés
  - F1-score : Moyenne harmonique de la précision et du rappel

- **Génération** :
  - Exact Match : Pourcentage de réponses identiques à la référence
  - Score BLEU : Évaluation de la qualité de la traduction
  - Score ROUGE : Mesure de similarité avec la référence

## 🤝 Contribution

1. Forkez le projet
2. Créez votre branche (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Poussez vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier `LICENSE` pour plus de détails.

## 📞 Contact

Pour toute question, veuillez ouvrir une issue sur le dépôt.
