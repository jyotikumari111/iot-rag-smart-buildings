from typing import List, Dict, Any

SYSTEM_POLICY = (
    "You are a factual, concise building-operations assistant. "
    "When making recommendations, cite manuals/specs using provided metadata (source + page_range). "
    "Use recent KPIs verbatim; do not fabricate data."
)

def format_context(ctxs: List[Dict[str, Any]]) -> str:
    out = []
    for c in ctxs:
        src = c.get("meta", {}).get("source", "N/A")
        pages = c.get("meta", {}).get("page_range", "N/A")
        out.append(f"[{src} p.{pages}]\n{c['text']}")
    return "\n\n".join(out)

def format_kpis(kpis: List[Dict[str, Any]]) -> str:
    if not kpis:
        return "No KPIs available in the last hour."
    lines = [f"{k['ts']}: mean={k['mean']:.3f}, std={k['std']:.3f}" for k in kpis]
    return "\n".join(lines)

def format_prompt(question: str, kpis: List[Dict[str, Any]], contexts: List[Dict[str, Any]]) -> str:
    ctx_blob = format_context(contexts)
    kpi_blob = format_kpis(kpis)
    return (
        f"{SYSTEM_POLICY}\n\n"
        f"Question: {question}\n\n"
        f"Recent KPIs (last hour):\n{kpi_blob}\n\n"
        f"Retrieved Contexts:\n{ctx_blob}\n\n"
        f"Instructions:\n"
        f"- Provide a clear, step-by-step recommendation.\n"
        f"- Quote thresholds/spec values from context when available.\n"
        f"- Cite sources like [source p.pages]."
    )
