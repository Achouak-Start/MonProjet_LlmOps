from sentence_transformers import CrossEncoder

NOM_MODELE_CROSSENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _reclasser_passages(requete: str, candidats: list[dict], modele_crossencoder, top_k_final: int = 3) -> list[dict]:
    """
    Reclasse une liste de candidats (résultats du bi-encoder) avec un cross-encoder,
    plus précis mais plus lent.

    candidats : liste de dicts avec au moins la clé "texte"
    Retourne : les top_k_final meilleurs candidats, avec un nouveau champ "score_rerank"
    """
    if not candidats:
        return []

    # 1. Construire les paires (requête, texte_candidat) attendues par le cross-encoder
    paires = [(requete, candidat["texte"]) for candidat in candidats]

    # 2. Calculer les scores de pertinence pour chaque paire
    scores = modele_crossencoder.predict(paires)

    # 3. Attacher le score à chaque candidat
    for candidat, score in zip(candidats, scores):
        candidat["score_rerank"] = float(score)

    # 4. Trier par score décroissant (le plus pertinent en premier)
    candidats_tries = sorted(candidats, key=lambda c: c["score_rerank"], reverse=True)

    # 5. Garder les top_k_final meilleurs
    return candidats_tries[:top_k_final]


if __name__ == "__main__":
    import chromadb
    from sentence_transformers import SentenceTransformer
    from rag.recherche import _rechercher_documents

    client = chromadb.PersistentClient(path="data/chroma_db")
    collection = client.get_collection("faq_service_client")
    modele_embedding = SentenceTransformer("all-MiniLM-L6-v2")
    modele_crossencoder = CrossEncoder(NOM_MODELE_CROSSENCODER)

    requete_test = "Je veux renvoyer un article, comment faire ?"

    # Étape 1 : bi-encoder récupère un nombre élargi de candidats
    candidats = _rechercher_documents(requete_test, collection, modele_embedding, top_k=10)
    print("Avant reranking :")
    for c in candidats:
        print(f"  [{c['id']}] score_biencoder={c['score']} → {c['texte'][:50]}...")

    # Étape 2 : cross-encoder reclasse et garde les meilleurs
    resultats_reranked = _reclasser_passages(requete_test, candidats, modele_crossencoder, top_k_final=3)
    print("\nAprès reranking :")
    for c in resultats_reranked:
        print(f"  [{c['id']}] score_rerank={c['score_rerank']:.3f} → {c['texte'][:50]}...")