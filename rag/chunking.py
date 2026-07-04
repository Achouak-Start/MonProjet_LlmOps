import json
from pathlib import Path


def _decouper_document(texte: str, taille_chunk: int = 200, chevauchement: int = 30) -> list[str]:
    """
    Découpe un texte en une liste de chunks (morceaux) de taille_chunk mots,
    avec un chevauchement de chevauchement mots entre chunks consécutifs.
    """
    mots = texte.split()

    if not mots:
        return []

    # Cas limite : le texte est plus court que la taille d'un chunk
    if len(mots) <= taille_chunk:
        return [" ".join(mots)]

    chunks = []
    # L'incrément entre deux fenêtres = taille_chunk - chevauchement
    increment = taille_chunk - chevauchement
    if increment <= 0:
        increment = taille_chunk  # sécurité si chevauchement >= taille_chunk

    debut = 0
    while debut < len(mots):
        fin = debut + taille_chunk
        chunk = mots[debut:fin]
        chunks.append(" ".join(chunk))
        if fin >= len(mots):
            break
        debut += increment

    return chunks


def _indexer_documents_longs(chemin_jsonl: Path, collection, modele_embedding, taille_chunk: int = 200, chevauchement: int = 30):
    """
    Charge des documents longs depuis un JSONL, les découpe en chunks,
    et les insère dans ChromaDB avec l'ID du document parent en métadonnée.
    """
    with open(chemin_jsonl, "r", encoding="utf-8") as f:
        documents = [json.loads(ligne) for ligne in f if ligne.strip()]

    ids_chunks = []
    textes_chunks = []
    metadatas_chunks = []

    for doc in documents:
        chunks = _decouper_document(doc["texte"], taille_chunk, chevauchement)
        for i, chunk in enumerate(chunks):
            ids_chunks.append(f"{doc['id']}__chunk_{i}")
            textes_chunks.append(chunk)
            metadatas_chunks.append({"document_parent": doc["id"]})

    if not ids_chunks:
        print("Aucun chunk à indexer.")
        return

    embeddings = modele_embedding.encode(textes_chunks).tolist()

    collection.add(
        ids=ids_chunks,
        embeddings=embeddings,
        documents=textes_chunks,
        metadatas=metadatas_chunks,
    )

    print(f"{len(ids_chunks)} chunks indexés depuis {len(documents)} document(s).")


#Test Chunking
if __name__ == "__main__":
    texte_exemple = "mot " * 100  # 100 mots factices pour tester
    chunks = _decouper_document(texte_exemple, taille_chunk=30, chevauchement=5)
    print(f"Nombre de chunks : {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"Chunk {i} : {len(c.split())} mots")