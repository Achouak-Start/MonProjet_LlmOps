import chromadb
from sentence_transformers import SentenceTransformer

from rag.recherche import _rechercher_documents
from rag.pipeline import _construire_prompt_augmente, _generer_avec_rag
from src.modele import charger_modele_et_tokeniseur

import pytest


@pytest.fixture(scope="module")
def collection_test():
    """Crée une collection ChromaDB temporaire en mémoire, peuplée de documents de test."""
    client = chromadb.Client()  # client en mémoire (pas persistant)
    collection = client.create_collection(name="test_faq", metadata={"hnsw:space": "cosine"})

    modele = SentenceTransformer("all-MiniLM-L6-v2")

    documents_test = [
        {"id": "faq-01", "question": "Quel est le délai de retour ?", "reponse": "30 jours."},
        {"id": "faq-02", "question": "Comment suivre ma commande ?", "reponse": "Par email."},
    ]

    questions = [doc["question"] for doc in documents_test]
    embeddings = modele.encode(questions).tolist()

    collection.add(
        ids=[doc["id"] for doc in documents_test],
        embeddings=embeddings,
        documents=[doc["reponse"] for doc in documents_test],
        metadatas=[{"question": doc["question"]} for doc in documents_test],
    )

    return collection, modele


def test_recherche_retourne_resultats(collection_test):
    collection, modele = collection_test
    resultats = _rechercher_documents("Comment retourner un article ?", collection, modele, top_k=2)
    assert len(resultats) > 0


def test_recherche_pertinence(collection_test):
    collection, modele = collection_test
    resultats = _rechercher_documents("Je veux retourner un article", collection, modele, top_k=2)
    ids_trouves = [r["id"] for r in resultats]
    assert "faq-01" in ids_trouves


def test_recherche_hors_domaine(collection_test):
    collection, modele = collection_test
    resultats = _rechercher_documents(
        "Quelle est la météo aujourd'hui ?", collection, modele, top_k=2
    )
    # Vérifie que les scores sont bas, plutôt que d'exiger zéro résultat
    for res in resultats:
        assert res["score"] < 0.55


def test_prompt_augmente_contient_contexte():
    documents = [{"id": "faq-01", "texte": "30 jours.", "score": 0.9, "metadata": {}}]
    prompt = _construire_prompt_augmente("Quel est le délai ?", documents)
    assert "30 jours." in prompt


def test_pipeline_complet_retourne_str(collection_test):
    collection, modele = collection_test
    modele_llm, tokeniseur = charger_modele_et_tokeniseur()
    resultat = _generer_avec_rag("Comment suivre ma commande ?", collection, modele, modele_llm, tokeniseur)
    assert isinstance(resultat["reponse"], str)
    assert len(resultat["reponse"]) > 0