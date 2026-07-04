from sentence_transformers import SentenceTransformer

NOM_MODELE_EMBEDDING = "all-MiniLM-L6-v2"
SEUIL_SIMILARITE = 0.4  # en dessous de ce score, on considère le doc non pertinent


def _rechercher_documents(requete: str, collection, modele_embedding, top_k: int = 3) -> list[dict]:
    """
    Recherche les documents les plus proches sémantiquement d'une requête.

    Retourne une liste de dicts : {"texte": ..., "score": ..., "id": ..., "metadata": ...}
    """
    # 1. Encoder la requête utilisateur en vecteur
    embedding_requete = modele_embedding.encode([requete]).tolist()

    # 2. Interroger ChromaDB pour les top_k documents les plus proches
    resultats = collection.query(
        query_embeddings=embedding_requete,
        n_results=top_k,
    )

    # 3. Construire la liste de résultats avec score de similarité
    documents_trouves = []
    ids = resultats["ids"][0]
    textes = resultats["documents"][0]
    distances = resultats["distances"][0]
    metadatas = resultats["metadatas"][0]

    for id_doc, texte, distance, metadata in zip(ids, textes, distances, metadatas):
        # Avec la distance cosinus, similarité = 1 - distance
        score_similarite = 1 - distance

        # On filtre les résultats sous le seuil minimum de pertinence
        if score_similarite >= SEUIL_SIMILARITE:
            documents_trouves.append({
                "id": id_doc,
                "texte": texte,
                "score": round(score_similarite, 3),
                "metadata": metadata,
            })

    return documents_trouves


if __name__ == "__main__":
    import chromadb

    client = chromadb.PersistentClient(path="data/chroma_db")
    collection = client.get_collection("faq_service_client")
    modele = SentenceTransformer(NOM_MODELE_EMBEDDING)

    # Test avec une question pertinente
    requete_test = "Je veux renvoyer un article, comment faire ?"
    resultats = _rechercher_documents(requete_test, collection, modele, top_k=3)

    print(f"Requête : {requete_test}\n")
    for res in resultats:
        print(f"[{res['id']}] score={res['score']} → {res['texte']}")