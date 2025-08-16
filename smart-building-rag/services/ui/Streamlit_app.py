import os, glob, re
import streamlit as st
import numpy as np

# Optional LLM (only used if OPENAI_API_KEY is set)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
USE_LLM = bool(OPENAI_API_KEY)

# Embeddings & lexical scoring
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

try:
    import fitz  # pymupdf for PDFs
except Exception:
    fitz = None

# ---------- Config ----------
DOCS_DIR = os.getenv("DOCS_DIR", "smart-building-rag/data/manuals")

# ---------- Utilities ----------
@st.cache_resource
def get_models():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model

def extract_text(path: str):
    if path.lower().endswith(".pdf"):
        if fitz is None:
            return "", []
        doc = fitz.open(path)
        pages = []
        for i in range(len(doc)):
            text = doc[i].get_text("text")
            pages.append((i+1, text))
        full = "\n".join(p for _, p in pages)
        return full, pages
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        return txt, [(1, txt)]

def tokenize_words(t: str):
    return re.findall(r"\w+|[^\w\s]", t)

def detokenize_words(tokens):
    return " ".join(tokens)

def chunk_text(text: str, target_tokens: int = 800, overlap: int = 120):
    toks = tokenize_words(text)
    if not toks: return []
    chunks, start = [], 0
    while start < len(toks):
        end = min(len(toks), start + target_tokens)
        chunk = detokenize_words(toks[start:end])
        chunks.append(chunk)
        if end == len(toks): break
        start = max(0, end - overlap)
    return chunks

def load_docs(root: str):
    files = glob.glob(os.path.join(root, "**/*.*"), recursive=True)
    docs = []
    for f in files:
        if f.lower().endswith((".pdf", ".txt", ".md")):
            full, pages = extract_text(f)
            for pno, ptxt in pages:
                for ch in chunk_text(ptxt, 800, 120):
                    docs.append({
                        "text": ch,
                        "meta": {"source": os.path.basename(f), "page_range": str(pno)}
                    })
    return docs

@st.cache_resource
def build_index(root: str):
    # Create sample doc if directory empty
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    if len(glob.glob(os.path.join(root, "**/*.*"), recursive=True)) == 0:
        sample_path = os.path.join(root, "ahu_maintenance.txt")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(
                "AHU_03 Maintenance (OEM X)\n"
                "- Check supply fan for vibration and unusual noise.\n"
                "- If supply air temp < 10 C for 15+ minutes with fan power > 6 kW, inspect freeze stat and coil bypass.\n"
                "- Filter ΔP above 250 Pa suggests replacement.\n"
                "- Coil cleaning every 3 months or when ΔT deviates > 3 C from design.\n"
                "- Torque spec for access panel M6 bolts: 8 N·m.\n"
            )

    docs = load_docs(root)
    if not docs:
        return {"docs": [], "bm25": None, "emb": None}

    model = get_models()
    texts = [d["text"] for d in docs]
    tokens = [t.split() for t in texts]
    bm25 = BM25Okapi(tokens)
    # Dense embeddings
    embs = model.encode(texts, batch_size=64, show_progress_bar=False)
    # Normalize for cosine similarity
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    embs = embs / norms
    return {"docs": docs, "bm25": bm25, "emb": embs}

def rrf_fuse(dense_scores, bm25_scores, k=5):
    # ranks to reciprocal ranks
    dense_rank = np.argsort(-dense_scores)
    bm25_rank = np.argsort(-bm25_scores)
    fused = {}
    for i, idx in enumerate(dense_rank):
        fused[idx] = fused.get(idx, 0) + 1.0/(60+i+1)
    for i, idx in enumerate(bm25_rank):
        fused[idx] = fused.get(idx, 0) + 1.0/(60+i+1)
    # top-k indices
    fused_items = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
    return [idx for idx, _ in fused_items]

def retrieve(query: str, index):
    docs = index["docs"]
    if not docs: return []
    bm25 = index["bm25"]
    embs = index["emb"]

    model = get_models()
    q_emb = model.encode([query])[0]
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)

    dense_scores = embs.dot(q_emb)
    bm25_scores = np.array(bm25.get_scores(query.split()))
    top_idx = rrf_fuse(dense_scores, bm25_scores, k=5)

    results = []
    for i in top_idx:
        results.append({
            "text": docs[i]["text"],
            "score": float(dense_scores[i]),
            "meta": docs[i]["meta"]
        })
    return results

def call_llm(question: str, ctxs):
    if not USE_LLM:
        # Extractive fallback
        bullets = []
        for c in ctxs[:3]:
            src = c["meta"].get("source", "N/A")
            pages = c["meta"].get("page_range", "N/A")
            bullets.append(f"- [{src} p.{pages}] {c['text'][:300]}...")
        return "Extractive answer (no API key set):\n" + "\n".join(bullets)

    # OpenAI Chat (optional)
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    ctx_blob = "\n\n".join(f"[{c['meta'].get('source','?')} p.{c['meta'].get('page_range','?')}] {c['text']}" for c in ctxs)
    prompt = (
        "You are a concise building-operations assistant. "
        "Use the provided contexts and cite sources like [source p.page]. "
        f"\n\nQuestion: {question}\n\nContexts:\n{ctx_blob}\n\n"
        "Answer with steps, thresholds, and a short recommendation."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":"You are a helpful building-ops assistant."},
                  {"role":"user","content":prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# ---------- UI ----------
st.set_page_config(page_title="Smart Building RAG", layout="wide")
st.title("🏙️ Smart Building Assistant (Self-contained)")

with st.expander("📁 Data folder", expanded=False):
    st.write(f"Using documents from: `{DOCS_DIR}`")

index = build_index(DOCS_DIR)
st.caption(f"Indexed {len(index['docs'])} chunks.")

col1, col2, col3 = st.columns([1,1,2])
with col1:
    site = st.text_input("Site ID", value="BLDG_A")
with col2:
    equip = st.text_input("Equipment ID", value="AHU_03")
with col3:
    q = st.text_input("Ask a question", value="Why is AHU_03 tripping and what should I check?")

if st.button("Ask"):
    if not q.strip():
        st.warning("Enter a question.")
    else:
        ctxs = retrieve(q, index)
        answer = call_llm(q, ctxs)
        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved Contexts"):
            for c in ctxs:
                meta = c.get("meta", {})
                st.caption(f"{meta.get('source','?')} p.{meta.get('page_range','?')}")
                st.write(c.get("text",""))
