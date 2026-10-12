"""
Module pour la recherche sémantique dans les documents indexés.
"""
from typing import List, Dict, Any


from .vector_store import VectorStoreManager

def search_documents(
    query: str,
    k: int = 5,
    min_score: float = 0.0,
    **search_kwargs
) -> List[Dict[str, Any]]:
    """
    Effectue une recherche sémantique dans les documents indexés.
    
    Args:
        query: La question ou requête de recherche en langage naturel
        k: Nombre maximum de résultats à retourner
        min_score: Score minimum de similarité pour inclure un résultat (0.0 à 1.0)
        **search_kwargs: Arguments supplémentaires pour la recherche
        
    Returns:
        Une liste de dictionnaires contenant pour chaque résultat:
        - 'content': Le texte du chunk
        - 'metadata': Les métadonnées du document (source, page, etc.)
        - 'score': Le score de similarité (entre 0 et 1)
    """
    # Initialisation du gestionnaire de stockage vectoriel
    vector_store = VectorStoreManager()
    
    # Recherche des documents similaires
    results_with_scores = vector_store.search(query, k=k, **search_kwargs)
    
    # Formatage des résultats
    formatted_results = []
    
    for doc, score in results_with_scores:
        # Normalisation du score entre 0 et 1 si nécessaire
        # (certains modèles peuvent retourner des scores hors de cette plage)
        normalized_score = max(0.0, min(1.0, float(score)))
        
        if normalized_score < min_score:
            continue
            
        formatted_results.append({
            'content': doc.page_content,
            'metadata': doc.metadata,
            'score': normalized_score
        })
    
    return formatted_results

def format_search_results(results: List[Dict[str, Any]]) -> str:
    """
    Formate les résultats de recherche pour l'affichage.
    
    Args:
        results: Liste des résultats de recherche formatés
        
    Returns:
        Chaîne formatée avec les résultats
    """
    if not results:
        return "Aucun résultat trouvé."
    
    output = []
    
    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        source = metadata.get('source', 'Source inconnue')
        page = metadata.get('page', 'N/A')
        
        # Formatage du contenu pour l'affichage
        content = result['content']
        if len(content) > 300:
            content = content[:297] + '...'
            
        output.append(
            f"\n📄 Résultat {i} (Score: {result['score']:.3f})"
            f"\n📂 Source: {source}"
            f"\n📄 Page: {page}"
            f"\n📝 Extrait: {content}"
            f"\n{'─' * 50}"
        )
    
    return "\n".join(output)
