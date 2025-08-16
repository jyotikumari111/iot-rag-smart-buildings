import os
from typing import List, Dict, Any, Optional
from chromadb import HttpClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLL = os.getenv("CHROMA_COLLECTION", "manual_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
_model = SentenceTransformer(EMBED_MODEL)

# naive cache for BM25
_TEXT_CACHE: List[Dict[str, Any]] = []

def _load_cache():
    global _TEXT_CACHE
    if _TEXT_CACHE:
        return
    try:
        coll = _client.get_collection(COLL)
        # fetch in small batches
        # chroma python client doesn't expose list-all easily; for demo assume <2000 docs
        res = coll.query(query_texts=["*"], n_results=1000)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        _TEXT_CACHE = [{"text": d, "meta": m} for d, m in zip(docs, metas)]
    except Exception:
        _TEXT_CACHE = []

def dense_search(query: str, k: int = 8, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    emb = _model.encode([query])[0].tolist()
    res = _client.get_collection(COLL).query(query_embeddings=[emb], n_results=k, where=filters or {})
    docs = res.get("documents", [[]])[0]
    dists = res.get("distances", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return [{"text": t, "score": float(s), "meta": m} for t, s, m in zip(docs, dists, metas)]

def lexical_search(query: str, k: int = 20) -> List[Dict[str, Any]]:
    _load_cache()
    if not _TEXT_CACHE:
        return []
    bm25 = BM25Okapi([t["text"].split() for t in _TEXT_CACHE])
    scores = bm25.get_scores(query.split())
    idx = np.argsort(scores)[::-1][:k]
    return [{"text": _TEXT_CACHE[i]["text"], "score": float(scores[i]), "meta": _TEXT_CACHE[i]["meta"]} for i in idx]

def rrf(lists: List[List[Dict[str, Any]]], k: int = 8) -> List[Dict[str, Any]]:
    # Reciprocal Rank Fusion over provided lists
    R = {}
    for L in lists:
        for rank, d in enumerate(L, start=1):
            key = d["text"]
            R[key] = R.get(key, 0) + 1.0/(60+rank)
    # flatten and sort by fused score
    flat = [d for L in lists for d in L]
    flat.sort(key=lambda d: R.get(d["text"], 0), reverse=True)
    seen, out = set(), []
    for d in flat:
        if d["text"] in seen:
            continue
        out.append(d)
        seen.add(d["text"])
        if len(out) >= k:
            break
    return out

def hybrid_retrieve(query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    d = dense_search(query, k=8, filters=filters)
    l = lexical_search(query, k=20)
    return rrf([d, l], k=5)
