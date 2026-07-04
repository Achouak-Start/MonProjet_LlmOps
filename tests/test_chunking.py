import chromadb
from sentence_transformers import SentenceTransformer

from rag.chunking import _decouper_document, _indexer_documents_longs
import pytest


def test_decoupe_sans_chevauchement():
    texte = "mot " * 100  # 100 mots
    chunks = _decouper_document(texte, taille_chunk=30, chevauchement=0)
    assert len(chunks) == 4  # 100/30 arrondi -> 3 pleins + 1 partiel


def test_decoupe_avec_chevauchement():
    # On utilise des mots numérotés pour repérer précisément la transition
    mots = [f"mot{i}" for i in range(100)]
    texte = " ".join(mots)
    chunks = _decouper_document(texte, taille_chunk=30, chevauchement=5)

    # Les 5 derniers mots du chunk 0 doivent aussi être les 5 premiers du chunk 1
    fin_chunk_0 = chunks[0].split()[-5:]
    debut_chunk_1 = chunks[1].split()[:5]
    assert fin_chunk_0 == debut_chunk_1


def test_chunk_vide():
    chunks = _decouper_document("", taille_chunk=30, chevauchement=5)
    assert chunks == []


def test_texte_plus_court_que_chunk():
    texte = "mot " * 50  # 50 mots
    chunks = _decouper_document(texte, taille_chunk=200, chevauchement=20)
    assert len(chunks) == 1


def test_metadonnees_source():
    client = chromadb.Client()  # collection en mémoire, isolée
    collection = client.create_collection(name="test_chunking", metadata={"hnsw:space": "cosine"})
    modele = SentenceTransformer("all-MiniLM-L6-v2")

    # On simule un mini fichier JSONL en écrivant un fichier temporaire
    import json
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        json.dump({"id": "doc-test", "texte": "mot " * 250}, f)
        f.write("\n")
        chemin_temp = Path(f.name)

    _indexer_documents_longs(chemin_temp, collection, modele, taille_chunk=100, chevauchement=10)

    resultats = collection.get()
    assert len(resultats["ids"]) > 0
    for metadata in resultats["metadatas"]:
        assert metadata["document_parent"] == "doc-test"