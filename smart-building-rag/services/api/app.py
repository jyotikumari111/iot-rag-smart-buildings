import os, json, time
from typing import Optional, List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from rag.retriever import hybrid_retrieve
from rag.prompts import format_prompt
from rag.indexer import ensure_collections
from ml.features import fetch_kpis
from ml.anomalies import recent_alerts, evaluate_point

import uvicorn
import requests

load_dotenv()

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8001"))

app = FastAPI(title="Smart Building RAG API")

class AskRequest(BaseModel):
    question: str
    site_id: Optional[str] = None
    equip_id: Optional[str] = None

class AlertIn(BaseModel):
    site_id: str
    equip_id: str
    sensor: str
    ts: Optional[str] = None
    severity: str = "WARN"
    message: str

ALERTS: List[Dict[str, Any]] = []  # simple in-memory store for demo

def call_llm(prompt: str) -> str:
    # Try OpenAI first
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model=openai_model,
                messages=[{"role": "system", "content": "You are a building operations assistant."},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            pass

    # Try Ollama local model
    ollama = os.getenv("OLLAMA_HOST", "").strip()
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")
    if ollama:
        try:
            r = requests.post(f"{ollama}/api/generate", json={"model": ollama_model, "prompt": prompt, "stream": False}, timeout=30)
            if r.ok:
                return r.json().get("response", "").strip()
        except Exception as e:
            pass

    # Safe fallback
    return "Assistant summary (fallback):\n" + prompt[:1200]

@app.on_event("startup")
def _on_startup():
    ensure_collections()

@app.get("/health")
def health():
    return {"ok": True, "time": time.time()}

@app.post("/ask")
def ask(req: AskRequest):
    kpis = fetch_kpis(req.site_id, req.equip_id)
    ctxs = hybrid_retrieve(req.question, filters={"equip_id": req.equip_id} if req.equip_id else None)
    prompt = format_prompt(question=req.question, kpis=kpis, contexts=ctxs)
    answer = call_llm(prompt)
    return {"answer": answer, "contexts": ctxs, "kpis": kpis}

@app.get("/alerts")
def get_alerts(site_id: Optional[str] = None, equip_id: Optional[str] = None):
    items = [a for a in ALERTS if (site_id is None or a["site_id"]==site_id) and (equip_id is None or a["equip_id"]==equip_id)]
    return {"alerts": items + recent_alerts(site_id, equip_id)}

@app.post("/alerts")
def post_alert(a: AlertIn):
    item = a.dict()
    item["ts"] = item.get("ts") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ALERTS.append(item)
    return {"ok": True, "alert": item}

if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
