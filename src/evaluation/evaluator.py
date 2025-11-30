"""
Module d'évaluation pour le système RAG.
"""
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

class RAGEvaluator:
    """Classe pour évaluer les performances du système RAG."""
    
    def __init__(self, test_data_path: Optional[str] = None):
        """
        Initialise l'évaluateur avec des données de test.
        
        Args:
            test_data_path: Chemin vers le fichier JSON contenant les données de test
        """
        self.test_data = []
        if test_data_path:
            self.load_test_data(test_data_path)
    
    def load_test_data(self, file_path: str):
        """Charge les données de test depuis un fichier JSON."""
        with open(file_path, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
    
    def evaluate_retrieval(self, retrieved_docs: List[Dict], relevant_docs: List[Dict]) -> Dict[str, float]:
        """
        Évalue la pertinence des documents récupérés.
        
        Args:
            retrieved_docs: Liste des documents récupérés par le système
            relevant_docs: Liste des documents pertinents (ground truth)
            
        Returns:
            Dictionnaire contenant les métriques d'évaluation
        """
        if not retrieved_docs or not relevant_docs:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            
        # Extraction des IDs des documents
        retrieved_ids = {doc.get('id') for doc in retrieved_docs}
        relevant_ids = {doc.get('id') for doc in relevant_docs}
        
        # Calcul des métriques
        if not retrieved_ids:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            
        # Précision et rappel
        intersection = retrieved_ids.intersection(relevant_ids)
        precision = len(intersection) / len(retrieved_ids) if retrieved_ids else 0.0
        recall = len(intersection) / len(relevant_ids) if relevant_ids else 0.0
        
        # F1-score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "retrieved_count": len(retrieved_ids),
            "relevant_count": len(relevant_ids)
        }
    
    def evaluate_generation(self, generated_answer: str, reference_answer: str) -> Dict[str, float]:
        """
        Évalue la qualité de la réponse générée.
        
        Args:
            generated_answer: Réponse générée par le système
            reference_answer: Réponse de référence (ground truth)
            
        Returns:
            Dictionnaire contenant les métriques d'évaluation
        """
        # Tokenisation simple pour le calcul de similarité
        gen_tokens = set(generated_answer.lower().split())
        ref_tokens = set(reference_answer.lower().split())
        
        if not gen_tokens or not ref_tokens:
            return {"bleu": 0.0, "rouge_l": 0.0, "exact_match": 0.0}
        
        # Calcul des métriques de base
        common_tokens = gen_tokens.intersection(ref_tokens)
        precision = len(common_tokens) / len(gen_tokens) if gen_tokens else 0.0
        recall = len(common_tokens) / len(ref_tokens) if ref_tokens else 0.0
        
        # F1-score (similaire à ROUGE-L)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Exact match
        exact_match = 1.0 if generated_answer.strip().lower() == reference_answer.strip().lower() else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "exact_match": exact_match
        }
    
    def evaluate_end_to_end(self, test_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Évalue le système de bout en bout sur un jeu de test.
        
        Args:
            test_data: Données de test (optionnel, utilise self.test_data si None)
            
        Returns:
            Dictionnaire contenant les résultats d'évaluation
        """
        if test_data is None:
            test_data = self.test_data
            
        if not test_data:
            raise ValueError("Aucune donnée de test fournie")
        
        results = {
            "retrieval_metrics": {"precision": [], "recall": [], "f1_score": []},
            "generation_metrics": {"precision": [], "recall": [], "f1_score": [], "exact_match": []},
            "samples": []
        }
        
        for item in test_data:
            # Évaluation de la récupération
            retrieval_metrics = self.evaluate_retrieval(
                item.get("retrieved_docs", []),
                item.get("relevant_docs", [])
            )
            
            # Évaluation de la génération
            generation_metrics = self.evaluate_generation(
                item.get("generated_answer", ""),
                item.get("reference_answer", "")
            )
            
            # Stockage des résultats
            results["samples"].append({
                "question": item.get("question"),
                "retrieval_metrics": retrieval_metrics,
                "generation_metrics": generation_metrics
            })
            
            # Agrégation des métriques
            for metric in ["precision", "recall", "f1_score"]:
                results["retrieval_metrics"][metric].append(retrieval_metrics.get(metric, 0.0))
                results["generation_metrics"][metric].append(generation_metrics.get(metric, 0.0))
                
            results["generation_metrics"]["exact_match"].append(generation_metrics.get("exact_match", 0.0))
        
        # Calcul des moyennes
        for metric_type in ["retrieval_metrics", "generation_metrics"]:
            # Créer une copie des clés pour éviter de modifier le dictionnaire pendant l'itération
            metrics = list(results[metric_type].keys())
            for metric in metrics:
                if isinstance(results[metric_type][metric], list):  # Ne traiter que les listes
                    results[metric_type][f"avg_{metric}"] = np.mean(results[metric_type][metric])
        
        return results
    
    @staticmethod
    def save_results(results: Dict, output_path: str):
        """Sauvegarde les résultats d'évaluation dans un fichier JSON."""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Résultats enregistrés dans {output_path}")

# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple de données de test
    test_data = [
        {
            "question": "Quelle est la durée de rétractation légale ?",
            "retrieved_docs": [
                {"id": "doc1", "content": "La durée de rétractation est de 14 jours."},
                {"id": "doc2", "content": "Délai de livraison : 3-5 jours ouvrés."}
            ],
            "relevant_docs": [
                {"id": "doc1", "content": "La durée de rétractation est de 14 jours."},
                {"id": "doc3", "content": "Le droit de rétractation s'applique sous 14 jours."}
            ],
            "generated_answer": "La durée de rétractation est de 14 jours.",
            "reference_answer": "Le délai de rétractation légal est de 14 jours."
        }
    ]
    
    # Initialisation et évaluation
    evaluator = RAGEvaluator()
    results = evaluator.evaluate_end_to_end(test_data)
    
    # Affichage des résultats
    print("\n=== Résultats d'évaluation ===")
    print("Métriques de récupération :")
    print(f"- Précision moyenne: {results['retrieval_metrics']['avg_precision']:.2f}")
    print(f"- Rappel moyen: {results['retrieval_metrics']['avg_recall']:.2f}")
    print(f"- F1-score moyen: {results['retrieval_metrics']['avg_f1_score']:.2f}")
    
    print("\nMétriques de génération :")
    print(f"- Précision moyenne: {results['generation_metrics']['avg_precision']:.2f}")
    print(f"- Rappel moyen: {results['generation_metrics']['avg_recall']:.2f}")
    print(f"- F1-score moyen: {results['generation_metrics']['avg_f1_score']:.2f}")
    print(f"- Exact match: {results['generation_metrics']['avg_exact_match']:.2f}")
