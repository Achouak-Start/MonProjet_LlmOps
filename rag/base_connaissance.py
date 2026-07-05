import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHEMIN_FAQ = Path("data/faq_service_client.jsonl")
CHEMIN_CHROMA_DB = "data/chroma_db"
NOM_COLLECTION = "faq_service_client"
NOM_MODELE_EMBEDDING = "all-MiniLM-L6-v2"


def charger_faq(chemin: Path) -> list[dict]:
    """Charge les entrées FAQ depuis un fichier JSONL."""
    documents = []
    with open(chemin, "r", encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                documents.append(json.loads(ligne))
    return documents


def creer_base_connaissance() -> chromadb.Collection:
    """Crée (ou ouvre) la collection ChromaDB et l'alimente avec la FAQ."""
    client = chromadb.PersistentClient(path=CHEMIN_CHROMA_DB)

    collection = client.get_or_create_collection(
        name=NOM_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # Évite les doublons si la collection est déjà peuplée
    if collection.count() > 0:
        print(
            f"Collection déjà peuplée ({collection.count()} documents). Aucune insertion."
        )
        return collection

    documents_faq = charger_faq(CHEMIN_FAQ)

    modele = SentenceTransformer(NOM_MODELE_EMBEDDING)

    # On encode la QUESTION (pas la réponse) — voir question de réflexion
    questions = [doc["question"] for doc in documents_faq]
    embeddings = modele.encode(questions).tolist()

    ids = [doc["id"] for doc in documents_faq]
    textes = [doc["reponse"] for doc in documents_faq]
    metadonnees = [
        {"question": doc["question"], "categorie": doc["categorie"]}
        for doc in documents_faq
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=textes,
        metadatas=metadonnees,
    )

    print(f"{len(ids)} documents insérés dans la collection '{NOM_COLLECTION}'.")
    return collection


if __name__ == "__main__":
    creer_base_connaissance()
