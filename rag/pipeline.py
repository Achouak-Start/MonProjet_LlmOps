import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag.recherche import _rechercher_documents
from rag.reranking import NOM_MODELE_CROSSENCODER, _reclasser_passages
from src.modele import charger_modele_et_tokeniseur, generer_reponse

PROMPT_SYSTEME = """Tu es un assistant service client.
Réponds uniquement en te basant sur le contexte fourni.
Si la réponse n'est pas dans le contexte, réponds : "Je n'ai pas l'information."

Contexte :
{contexte}

Question : {question}

Réponse :"""

TOP_K_BIENCODER = 10
TOP_K_FINAL = 3


def _construire_prompt_augmente(question: str, documents: list[dict]) -> str:
    """Construit le prompt augmenté en injectant les documents trouvés."""
    if not documents:
        contexte = "(aucun document pertinent trouvé)"
    else:
        contexte = "\n".join(f"[{i+1}] {doc['texte']}" for i, doc in enumerate(documents))

    return PROMPT_SYSTEME.format(contexte=contexte, question=question)


def _generer_avec_rag(
    prompt_utilisateur: str,
    collection,
    modele_embedding,
    modele_llm,
    tokeniseur,
    modele_crossencoder=None,
) -> dict:
    """
    Pipeline RAG complet : recherche (bi-encoder) -> reranking
    (cross-encoder) -> prompt augmenté -> génération.
    Retourne : {"reponse": ..., "documents_sources": [ids des documents utilisés]}
    """
    # 1. Recherche élargie des documents candidats (bi-encoder)
    documents_trouves = _rechercher_documents(
        prompt_utilisateur, collection, modele_embedding, top_k=TOP_K_BIENCODER
    )

    # 2. Cas dégénéré : aucun document pertinent trouvé
    if not documents_trouves:
        return {
            "reponse": "Je n'ai pas l'information.",
            "documents_sources": [],
        }

    # 3. Reranking avec le cross-encoder (si fourni)
    if modele_crossencoder is not None:
        documents_trouves = _reclasser_passages(
            prompt_utilisateur,
            documents_trouves,
            modele_crossencoder,
            top_k_final=TOP_K_FINAL,
        )
    else:
        documents_trouves = documents_trouves[:TOP_K_FINAL]

    # 4. Construction du prompt augmenté
    prompt_augmente = _construire_prompt_augmente(prompt_utilisateur, documents_trouves)

    # 5. Génération avec le LLM (à partir du prompt AUGMENTÉ, pas l'original)
    reponse = generer_reponse(prompt_augmente, modele_llm, tokeniseur)

    # 6. Traçabilité des sources utilisées
    ids_sources = [doc["id"] for doc in documents_trouves]

    return {
        "reponse": reponse,
        "documents_sources": ids_sources,
    }


if __name__ == "__main__":
    client = chromadb.PersistentClient(path="data/chroma_db")
    collection = client.get_collection("faq_service_client")
    modele_embedding = SentenceTransformer("all-MiniLM-L6-v2")
    modele_crossencoder = CrossEncoder(NOM_MODELE_CROSSENCODER)

    print("Chargement du LLM (peut prendre 10-30 secondes)...")
    modele_llm, tokeniseur = charger_modele_et_tokeniseur()

    question_test = "Je veux renvoyer un article, comment faire ?"
    resultat = _generer_avec_rag(
        question_test,
        collection,
        modele_embedding,
        modele_llm,
        tokeniseur,
        modele_crossencoder,
    )

    print(f"\nQuestion : {question_test}")
    print(f"Réponse : {resultat['reponse']}")
    print(f"Sources : {resultat['documents_sources']}")
