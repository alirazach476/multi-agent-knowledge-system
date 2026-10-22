"""Streamlit UI for the multi-agent knowledge system (GPU embeddings + live trace)."""

from __future__ import annotations

import html
import re

import streamlit as st

from multi_agent import (
    ConfigurationError,
    ask_team_stream,
    ensure_vectorstore,
    get_chat_model_name,
    get_device_label,
    get_embed_model_name,
    list_knowledge_files,
    require_api_key,
    warm_up,
)

EXAMPLE_QUESTIONS = [
    "How do I get API access and how often do keys expire?",
    "What retrieval strategy and chunk sizes do we use?",
    "Where are audit logs and how long are they retained?",
    "What is the on-call escalation path for a production RAG outage?",
]

AGENT_META = {
    "supervisor": {"label": "Supervisor", "color": "#0F766E", "role": "Routes work"},
    "researcher": {"label": "Researcher", "color": "#1D4ED8", "role": "Retrieves context"},
    "analyst": {"label": "Analyst", "color": "#B45309", "role": "Synthesizes facts"},
    "writer": {"label": "Writer", "color": "#0E7490", "role": "Drafts answer"},
}

PAGE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Fraunces:opsz,wght@9..144,560;9..144,650&display=swap');

:root {
  --ink: #0B1220;
  --ink-soft: #334155;
  --muted: #64748B;
  --line: #E2E8F0;
  --panel: rgba(255,255,255,0.78);
  --panel-solid: #FFFFFF;
  --accent: #0F766E;
  --accent-deep: #115E59;
  --accent-soft: #CCFBF1;
  --bg0: #F1F5F9;
  --bg1: #E8EEF6;
  --shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
}

html, body, [class*="css"] {
  font-family: "DM Sans", sans-serif;
}

.stApp {
  background:
    radial-gradient(1200px 600px at 8% -10%, #CFFAFE 0%, transparent 55%),
    radial-gradient(900px 500px at 100% 0%, #DBEAFE 0%, transparent 50%),
    linear-gradient(180deg, var(--bg1) 0%, var(--bg0) 42%, #F8FAFC 100%);
  color: var(--ink);
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none; }
#MainMenu, footer { visibility: hidden; }

.block-container {
  padding-top: 1.4rem !important;
  padding-bottom: 3rem !important;
  max-width: 1120px;
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0B1220 0%, #111827 55%, #0F172A 100%);
  border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] label { color: #94A3B8 !important; }
section[data-testid="stSidebar"] hr {
  border-color: rgba(148,163,184,0.25) !important;
}
section[data-testid="stSidebar"] .stButton > button {
  background: rgba(15, 118, 110, 0.18) !important;
  border: 1px solid rgba(45, 212, 191, 0.35) !important;
  color: #99F6E4 !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(15, 118, 110, 0.35) !important;
  border-color: #2DD4BF !important;
}

.brand-mark {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  margin-bottom: 0.85rem;
  animation: rise 0.55s ease both;
}
.brand-orb {
  width: 14px;
  height: 14px;
  border-radius: 4px;
  background: linear-gradient(135deg, #2DD4BF, #0F766E);
  box-shadow: 0 0 0 4px rgba(15,118,110,0.15);
}
.brand-name {
  font-family: "Fraunces", serif;
  font-size: 1.05rem;
  font-weight: 650;
  letter-spacing: -0.02em;
  color: var(--ink);
}
.hero-title {
  font-family: "Fraunces", serif;
  font-size: clamp(2rem, 4vw, 2.65rem);
  font-weight: 650;
  letter-spacing: -0.03em;
  line-height: 1.12;
  color: var(--ink);
  margin: 0 0 0.65rem 0;
  animation: rise 0.65s ease 0.05s both;
}
.hero-sub {
  max-width: 38rem;
  color: var(--ink-soft);
  font-size: 1.05rem;
  line-height: 1.55;
  margin: 0 0 1.1rem 0;
  animation: rise 0.65s ease 0.12s both;
}
.pill-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin-bottom: 1.35rem;
  animation: rise 0.65s ease 0.18s both;
}
.pill {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.28rem 0.7rem;
  border-radius: 999px;
  background: rgba(255,255,255,0.7);
  border: 1px solid var(--line);
  color: var(--ink-soft);
  font-size: 0.8rem;
  font-weight: 600;
}
.pill .dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 0 3px rgba(15,118,110,0.15);
  animation: pulse 2.2s ease infinite;
}
.ask-shell {
  background: var(--panel);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(226,232,240,0.9);
  border-radius: 18px;
  padding: 1.1rem 1.15rem 1.2rem;
  box-shadow: var(--shadow);
  margin-bottom: 1.25rem;
  animation: rise 0.7s ease 0.2s both;
}
.ask-label {
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.55rem;
}
.panel {
  background: var(--panel-solid);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 1rem 1.05rem 1.1rem;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
  min-height: 120px;
}
.panel-title {
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.75rem;
}
.answer-body {
  color: var(--ink);
  font-size: 1.02rem;
  line-height: 1.65;
}
.trace-item {
  position: relative;
  border-left: 3px solid #CBD5E1;
  padding: 0.55rem 0.8rem 0.55rem 0.85rem;
  margin: 0.45rem 0;
  background: linear-gradient(90deg, #F8FAFC 0%, #FFFFFF 100%);
  border-radius: 0 12px 12px 0;
  font-size: 0.9rem;
  color: var(--ink-soft);
  animation: slideIn 0.28s ease both;
}
.trace-item strong {
  display: block;
  font-size: 0.82rem;
  letter-spacing: 0.02em;
  margin-bottom: 0.15rem;
}
.side-brand {
  font-family: "Fraunces", serif;
  font-size: 1.2rem;
  font-weight: 650;
  color: #F8FAFC !important;
  margin-bottom: 0.2rem;
}
.side-tag {
  color: #94A3B8 !important;
  font-size: 0.82rem !important;
  margin-bottom: 1rem;
}
.meta-grid {
  display: grid;
  gap: 0.55rem;
  margin: 0.85rem 0 1rem;
}
.meta-row {
  display: flex;
  justify-content: space-between;
  gap: 0.75rem;
  padding: 0.55rem 0.65rem;
  border-radius: 10px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(148,163,184,0.12);
  font-size: 0.8rem;
}
.meta-row span:first-child { color: #94A3B8 !important; }
.meta-row span:last-child {
  color: #F1F5F9 !important;
  font-weight: 600;
  text-align: right;
  word-break: break-all;
}
.status-ok { color: #5EEAD4 !important; font-weight: 650; font-size: 0.88rem; }
.status-bad { color: #FDA4AF !important; font-weight: 650; font-size: 0.88rem; }
.gpu-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  margin-top: 0.35rem;
  padding: 0.28rem 0.65rem;
  border-radius: 999px;
  background: rgba(45, 212, 191, 0.12);
  border: 1px solid rgba(45, 212, 191, 0.35);
  color: #99F6E4 !important;
  font-size: 0.78rem;
  font-weight: 650;
}
.empty-hint {
  border: 1px dashed #CBD5E1;
  border-radius: 14px;
  padding: 1.1rem 1.2rem;
  color: var(--muted);
  background: rgba(255,255,255,0.45);
}
div.stButton > button[kind="primary"] {
  background: linear-gradient(180deg, #0F766E 0%, #0D9488 100%) !important;
  border: none !important;
  color: white !important;
  border-radius: 12px !important;
  font-weight: 650 !important;
  padding: 0.55rem 1.1rem !important;
  box-shadow: 0 10px 24px rgba(15, 118, 110, 0.28);
}
div.stButton > button[kind="primary"]:hover {
  filter: brightness(1.05);
}
div.stButton > button[kind="secondary"] {
  border-radius: 999px !important;
  border: 1px solid var(--line) !important;
  background: rgba(255,255,255,0.75) !important;
  color: var(--ink-soft) !important;
  font-size: 0.82rem !important;
  font-weight: 550 !important;
}
@keyframes rise {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes slideIn {
  from { opacity: 0; transform: translateX(-8px); }
  to { opacity: 1; transform: translateX(0); }
}
@keyframes pulse {
  0%, 100% { box-shadow: 0 0 0 3px rgba(15,118,110,0.12); }
  50% { box-shadow: 0 0 0 6px rgba(15,118,110,0.08); }
}
"""


def _parse_step(step: str) -> tuple[str, str]:
    lower = step.lower()
    for key in AGENT_META:
        if key in lower:
            return key, step
    return "supervisor", step


def _short_model(name: str) -> str:
    return name.split("/")[-1]


def _render_trace_html(steps: list[str]) -> str:
    parts: list[str] = []
    for i, step in enumerate(steps):
        key, text = _parse_step(step)
        meta = AGENT_META[key]
        safe = html.escape(text)
        # Drop redundant agent prefix in body when present
        body = re.sub(rf"^{key}\s*", "", safe, flags=re.I)
        parts.append(
            f'<div class="trace-item" style="border-left-color:{meta["color"]};'
            f'animation-delay:{i * 0.04:.2f}s">'
            f'<strong style="color:{meta["color"]}">{meta["label"]}</strong>'
            f"{body}</div>"
        )
    return "".join(parts) or '<div class="empty-hint">Waiting for the first agent hop…</div>'


def _render_answer_html(text: str) -> str:
    safe = html.escape(text or "").replace("\n", "<br>")
    return f'<div class="panel"><div class="panel-title">Final answer</div>' f'<div class="answer-body">{safe}</div></div>'


@st.cache_resource(show_spinner="Warming embeddings + FAISS…")
def _cached_warm_up() -> dict[str, str]:
    return warm_up()


st.set_page_config(
    page_title="Aether Knowledge · Multi-Agent RAG",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"<style>{PAGE_CSS}</style>", unsafe_allow_html=True)

docs = list_knowledge_files()
device_label = get_device_label()
chat_model = get_chat_model_name()
embed_model = get_embed_model_name()

try:
    require_api_key()
    api_ok = True
except ConfigurationError:
    api_ok = False

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="side-brand">Aether Knowledge</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-tag">Multi-agent RAG console</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="meta-grid">
          <div class="meta-row"><span>Provider</span><span>Groq</span></div>
          <div class="meta-row"><span>Chat</span><span>{html.escape(_short_model(chat_model))}</span></div>
          <div class="meta-row"><span>Embeddings</span><span>{html.escape(_short_model(embed_model))}</span></div>
          <div class="meta-row"><span>Corpus</span><span>{len(docs)} docs</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="gpu-badge">● {html.escape(device_label)}</span>',
        unsafe_allow_html=True,
    )

    if docs:
        with st.expander("Knowledge files", expanded=False):
            for path in docs:
                st.write(f"`{path.name}`")

    st.markdown("")
    if api_ok:
        st.markdown('<p class="status-ok">API connected</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-bad">API key missing</p>', unsafe_allow_html=True)
        st.caption("Add GROQ_API_KEY in .env")

    st.divider()
    if st.button("Rebuild index", use_container_width=True):
        try:
            with st.spinner("Rebuilding FAISS…"):
                _cached_warm_up.clear()
                ensure_vectorstore(rebuild=True)
                _cached_warm_up()
            st.success("Index rebuilt.")
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

    if st.button("Warm engine", use_container_width=True):
        try:
            info = _cached_warm_up()
            st.success(f"Ready · {info['device']}")
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

    st.caption("Embeddings: local CUDA/CPU · Chat: Groq API")

try:
    warm_info = _cached_warm_up()
except Exception:  # noqa: BLE001
    warm_info = None

# --- Main ---
st.markdown(
    """
    <div class="brand-mark">
      <span class="brand-orb"></span>
      <span class="brand-name">Aether Knowledge</span>
    </div>
    <h1 class="hero-title">Ask your knowledge team.</h1>
    <p class="hero-sub">
      A supervisor routes each question through researcher, analyst, and writer agents —
      grounded in your local docs, accelerated on GPU when available.
    </p>
    """,
    unsafe_allow_html=True,
)

ready_text = f"Engine ready · {warm_info['device']}" if warm_info else "Engine idle"
st.markdown(
    f"""
    <div class="pill-row">
      <span class="pill"><span class="dot"></span>{html.escape(ready_text)}</span>
      <span class="pill">Live agent trace</span>
      <span class="pill">Citation-aware answers</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []
if "question_input" not in st.session_state:
    st.session_state.question_input = ""

st.markdown('<div class="ask-label">Compose</div>', unsafe_allow_html=True)
st.markdown("**Quick prompts**")
qcols = st.columns(2)
for i, example in enumerate(EXAMPLE_QUESTIONS):
    with qcols[i % 2]:
        if st.button(example, key=f"ex_{i}", use_container_width=True):
            st.session_state.question_input = example
            st.rerun()

question = st.text_input(
    "Question",
    key="question_input",
    placeholder="Ask about API access, retention, incidents, SLA…",
    label_visibility="collapsed",
)
run = st.button("Run agent team", type="primary", disabled=not bool(question.strip()))

if run and question.strip():
    if not api_ok:
        st.error("Configure a valid GROQ_API_KEY in `.env` before running.")
    elif not docs:
        st.error("Add Markdown or text files under `knowledge/` then rebuild the index.")
    else:
        left, right = st.columns([1, 1.35], gap="large")
        with left:
            st.markdown('<div class="panel-title">Agent trace</div>', unsafe_allow_html=True)
            trace_box = st.empty()
            status = st.status("Agents collaborating…", expanded=True)
        with right:
            answer_box = st.empty()
            answer_box.markdown(
                '<div class="panel"><div class="panel-title">Final answer</div>'
                '<div class="answer-body" style="color:#64748B">Waiting for the writer…</div></div>',
                unsafe_allow_html=True,
            )
            detail_box = st.empty()

        result_state: dict = {}
        try:
            for event in ask_team_stream(question.strip()):
                node = event["node"]
                state = event["state"]
                result_state = state
                label = AGENT_META.get(node, {"label": node.title()})["label"]
                status.update(label=f"Running · {label}")
                status.write(f"✓ {label}")
                trace_box.markdown(_render_trace_html(state.get("steps") or []), unsafe_allow_html=True)
                if state.get("final_answer"):
                    answer_box.markdown(_render_answer_html(state["final_answer"]), unsafe_allow_html=True)
                notes = state.get("research_notes") or ""
                analysis = state.get("analysis") or ""
                if notes or analysis:
                    with detail_box.container():
                        if notes:
                            with st.expander("Research notes", expanded=False):
                                st.markdown(notes)
                        if analysis:
                            with st.expander("Analysis", expanded=False):
                                st.markdown(analysis)

            status.update(label="Complete", state="complete")
            st.session_state.history.insert(
                0,
                {"question": question.strip(), "result": result_state},
            )
        except ConfigurationError as exc:
            status.update(label="Failed", state="error")
            st.error(str(exc))
        except FileNotFoundError as exc:
            status.update(label="Failed", state="error")
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            status.update(label="Failed", state="error")
            st.error(f"Workflow failed: {exc}")

elif st.session_state.history:
    latest = st.session_state.history[0]
    result = latest["result"]

    left, right = st.columns([1, 1.35], gap="large")
    with left:
        st.markdown('<div class="panel-title">Agent trace</div>', unsafe_allow_html=True)
        st.markdown(_render_trace_html(result.get("steps") or []), unsafe_allow_html=True)
    with right:
        st.markdown(_render_answer_html(result.get("final_answer") or "No answer produced."), unsafe_allow_html=True)
        with st.expander("Research notes", expanded=False):
            st.markdown(result.get("research_notes") or "_Empty_")
        with st.expander("Analysis", expanded=False):
            st.markdown(result.get("analysis") or "_Empty_")

    if len(st.session_state.history) > 1:
        with st.expander(f"Earlier runs ({len(st.session_state.history) - 1})"):
            for item in st.session_state.history[1:]:
                st.markdown(f"**{item['question']}**")
                st.markdown(item["result"].get("final_answer") or "")
                st.divider()
else:
    st.markdown(
        '<div class="empty-hint">Pick a quick prompt or type a question — '
        "the live trace updates as each specialist finishes.</div>",
        unsafe_allow_html=True,
    )
