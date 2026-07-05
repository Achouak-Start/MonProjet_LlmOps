import chromadb
import pytest
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag.pipeline import _construire_prompt_augmente, _generer_avec_rag
from rag.recherche import _rechercher_documents
from rag.reranking import NOM_MODELE_CROSSENCODER, _reclasser_passages
from src.modele import charger_modele_et_tokeniseur


@pytest.fixture(scope="module")
def collection_test():
    """Crée une collection ChromaDB temporaire en mémoire, peuplée de documents de test."""
    client = chromadb.Client()  # client en mémoire (pas persistant)
    collection = client.create_collection(name="test_faq", metadata={"hnsw:space": "cosine"})

    modele = SentenceTransformer("all-MiniLM-L6-v2")

    documents_test = [
        {
            "id": "faq-01",
            "question": "Quel est le délai de retour ?",
            "reponse": "30 jours.",
        },
        {
            "id": "faq-02",
            "question": "Comment suivre ma commande ?",
            "reponse": "Par email.",
        },
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
    resultat = _generer_avec_rag(
        "Comment suivre ma commande ?", collection, modele, modele_llm, tokeniseur
    )
    assert isinstance(resultat["reponse"], str)
    assert len(resultat["reponse"]) > 0


@pytest.fixture(scope="module")
def modele_crossencoder_test():
    return CrossEncoder(NOM_MODELE_CROSSENCODER)


def test_reranking_retourne_top_k(modele_crossencoder_test):
    candidats = [
        {"id": "a", "texte": "Le délai de retour est de 30 jours."},
        {"id": "b", "texte": "La météo est ensoleillée aujourd'hui."},
        {"id": "c", "texte": "Vous pouvez retourner un article sous 30 jours."},
        {"id": "d", "texte": "Notre siège social est à Paris."},
    ]
    resultats = _reclasser_passages(
        "Quel est le délai de retour ?",
        candidats,
        modele_crossencoder_test,
        top_k_final=2,
    )
    assert len(resultats) == 2


def test_reranking_ordre_coherent(modele_crossencoder_test):
    candidats = [
        {
            "id": "pertinent",
            "texte": "Le délai de retour est de 30 jours pour tout article.",
        },
        {
            "id": "non_pertinent",
            "texte": "Nous vendons des cartes cadeaux de 10 à 200 euros.",
        },
    ]
    resultats = _reclasser_passages(
        "Quel est le délai de retour ?",
        candidats,
        modele_crossencoder_test,
        top_k_final=2,
    )
    assert resultats[0]["id"] == "pertinent"


def test_reranking_integre_pipeline(collection_test, modele_crossencoder_test):
    collection, modele = collection_test
    modele_llm, tokeniseur = charger_modele_et_tokeniseur()
    resultat = _generer_avec_rag(
        "Comment suivre ma commande ?",
        collection,
        modele,
        modele_llm,
        tokeniseur,
        modele_crossencoder_test,
    )
    assert isinstance(resultat["reponse"], str)
    assert len(resultat["reponse"]) > 0
