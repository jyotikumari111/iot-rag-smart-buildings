import os, json, requests
import streamlit as st

API_HOST = os.getenv("API_HOST", "api")
API_PORT = int(os.getenv("API_PORT", "8001"))
API_URL = f"http://{API_HOST}:{API_PORT}"

st.set_page_config(page_title="Smart Building RAG", layout="wide")
st.title("🏙️ Smart Building Assistant")

col1, col2, col3 = st.columns([1,1,2])
with col1:
    site = st.text_input("Site ID", value="BLDG_A")
with col2:
    equip = st.text_input("Equipment ID", value="AHU_03")
with col3:
    q = st.text_input("Ask a question", value="Why is AHU_03 tripping and what should I check?")

if st.button("Ask"):
    r = requests.post(f"{API_URL}/ask", json={"question": q, "site_id": site, "equip_id": equip}, timeout=60)
    data = r.json()
    st.subheader("Answer")
    st.write(data.get("answer", ""))
    with st.expander("Contexts"):
        for c in data.get("contexts", []):
            meta = c.get("meta", {})
            st.caption(f"{meta.get('source','?')} p.{meta.get('page_range','?')}")
            st.write(c.get("text",""))
    with st.expander("KPIs (last hour)"):
        st.json(data.get("kpis", []))

st.divider()
st.subheader("Alerts")
params = {}
if site: params["site_id"] = site
if equip: params["equip_id"] = equip
try:
    ar = requests.get(f"{API_URL}/alerts", params=params, timeout=15).json()
    for a in ar.get("alerts", []):
        st.write(f"**{a.get('severity','INFO')}** — {a.get('ts','')} — {a.get('message','')}")
except Exception as e:
    st.warning("Alert service not reachable yet.")
