import fitz  # pymupdf
import re
from typing import List, Tuple

def extract_text(path: str):
    if path.lower().endswith(".pdf"):
        doc = fitz.open(path)
        pages = []
        for i in range(len(doc)):
            text = doc[i].get_text("text")
            pages.append((i+1, text))
        full = "\n".join(p for _, p in pages)
        return full, pages
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    # treat as single page
    return txt, [(1, txt)]

def tokenize_words(t: str):
    return re.findall(r"\w+|\S", t)

def detokenize_words(tokens: List[str]):
    return " ".join(tokens)

def chunk_text(text: str, target_tokens: int = 800, overlap: int = 120) -> List[str]:
    toks = tokenize_words(text)
    if not toks:
        return []
    chunks = []
    start = 0
    while start < len(toks):
        end = min(len(toks), start + target_tokens)
        chunk = detokenize_words(toks[start:end])
        chunks.append(chunk)
        if end == len(toks):
            break
        start = max(0, end - overlap)
    return chunks
