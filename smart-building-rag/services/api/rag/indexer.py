import os, glob, uuid
from typing import List, Dict, Any, Tuple
from chromadb import HttpClient
from sentence_transformers import SentenceTransformer
from .utils import extract_text, chunk_text

CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLL = os.getenv("CHROMA_COLLECTION", "manual_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
_model = SentenceTransformer(EMBED_MODEL)

def ensure_collections():
    try:
        _client.get_collection(COLL)
    except Exception:
        _client.create_collection(COLL, metadata={"hnsw:space": "cosine"})

def index_file(path: str, source: str):
    text, pages = extract_text(path)  # pages: list of (page_no, text)
    chunks = []
    for pno, t in pages:
        for ch in chunk_text(t, target_tokens=800, overlap=120):
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": ch,
                "meta": {"source": source, "page_range": str(pno), "path": os.path.basename(path)}
            })
    if not chunks:
        return 0
    coll = _client.get_collection(COLL)
    docs = [c["text"] for c in chunks]
    embs = _model.encode(docs, batch_size=64, show_progress_bar=False)
    coll.add(documents=docs, embeddings=embs.tolist(), metadatas=[c["meta"] for c in chunks], ids=[c["id"] for c in chunks])
    return len(chunks)

def reindex_manuals(root: str = "/app/data/manuals"):
    ensure_collections()
    files = glob.glob(os.path.join(root, "**/*.*"), recursive=True)
    total = 0
    for f in files:
        if f.lower().endswith((".pdf", ".txt", ".md")):
            total += index_file(f, source=os.path.basename(f))
    print(f"Indexed {total} chunks from {len(files)} files.")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/app/data/manuals")
    args = p.parse_args()
    reindex_manuals(args.root)
