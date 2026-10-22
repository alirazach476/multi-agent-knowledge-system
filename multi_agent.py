"""
Multi-Agent Knowledge System

Supervisor routes to specialized agents:
- Researcher (RAG over docs)
- Analyst (structured synthesis)
- Writer (final answer)

Uses Groq (free API tier) for chat + local HuggingFace embeddings on GPU when available.
"""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Literal, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

BASE = Path(__file__).resolve().parent
DOCS = BASE / "knowledge"
VECTOR = BASE / "vectorstore"
MAX_SUPERVISOR_HOPS = 8
DEFAULT_CHAT_MODEL = "openai/gpt-oss-20b"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class ConfigurationError(RuntimeError):
    """Raised when required environment or data setup is missing."""


class RouteDecision(BaseModel):
    next_agent: Literal["researcher", "analyst", "writer", "finish"]
    reason: str = Field(description="Short rationale for the routing choice")


class TeamState(TypedDict):
    question: str
    route: str
    research_notes: str
    analysis: str
    final_answer: str
    steps: list[str]
    hop_count: int


def _secret(name: str, default: str = "") -> str:
    """Read from env first, then Streamlit Cloud secrets."""
    value = (os.getenv(name) or "").strip()
    if value:
        return value
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name]).strip()
        # Optional nested table: [groq] api_key = "..."
        if "groq" in st.secrets and name == "GROQ_API_KEY":
            nested = st.secrets["groq"]
            if "api_key" in nested:
                return str(nested["api_key"]).strip()
    except Exception:  # noqa: BLE001 — not running under Streamlit / no secrets
        pass
    return default


def detect_device() -> str:
    """Prefer CUDA when available; Streamlit Cloud has no GPU → CPU."""
    forced = _secret("EMBED_DEVICE").lower()
    if forced in {"cuda", "cpu", "mps"}:
        return forced
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def get_device_label() -> str:
    device = detect_device()
    if device != "cuda":
        return device.upper()
    try:
        import torch

        name = torch.cuda.get_device_name(0)
        return f"CUDA · {name}"
    except Exception:  # noqa: BLE001
        return "CUDA"


def require_api_key() -> str:
    key = _secret("GROQ_API_KEY")
    if not key or key.startswith("gsk_your"):
        raise ConfigurationError(
            "GROQ_API_KEY is missing. For local use, set it in .env. "
            "On Streamlit Cloud, add it under App settings → Secrets. "
            "Get a free key at https://console.groq.com/keys"
        )
    return key


def get_chat_model_name() -> str:
    return _secret("GROQ_MODEL", DEFAULT_CHAT_MODEL)


def get_embed_model_name() -> str:
    return _secret("EMBEDDING_MODEL", DEFAULT_EMBED_MODEL)


def list_knowledge_files() -> list[Path]:
    if not DOCS.exists():
        return []
    return sorted(
        p for p in DOCS.glob("**/*") if p.is_file() and p.suffix.lower() in {".txt", ".md"}
    )


def get_llm(temp: float = 0.2, *, max_tokens: int | None = None) -> ChatGroq:
    require_api_key()
    kwargs: dict[str, Any] = {
        "model": get_chat_model_name(),
        "temperature": temp,
        "groq_api_key": require_api_key(),
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatGroq(**kwargs)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Local embeddings on GPU when CUDA is available."""
    device = detect_device()
    encode_kwargs: dict[str, Any] = {
        "normalize_embeddings": True,
        "batch_size": 64 if device == "cuda" else 16,
    }
    embeddings = HuggingFaceEmbeddings(
        model_name=get_embed_model_name(),
        model_kwargs={"device": device},
        encode_kwargs=encode_kwargs,
        show_progress=False,
    )
    # FP16 on CUDA for faster encode on Quadro / consumer GPUs
    if device == "cuda":
        try:
            embeddings.client = embeddings.client.half()
        except Exception:  # noqa: BLE001
            pass
    return embeddings


def _clear_vector_cache() -> None:
    get_cached_vectorstore.cache_clear()


@lru_cache(maxsize=1)
def get_cached_vectorstore() -> FAISS:
    return _load_or_build_vectorstore()


def _load_or_build_vectorstore() -> FAISS:
    embeddings = get_embeddings()

    if (VECTOR / "index.faiss").exists():
        # Local FAISS pickle is trusted because we build it ourselves.
        return FAISS.load_local(
            str(VECTOR), embeddings, allow_dangerous_deserialization=True
        )

    files = list_knowledge_files()
    if not files:
        raise FileNotFoundError(
            f"No knowledge documents found. Add .md or .txt files under {DOCS}"
        )

    docs = []
    for path in files:
        docs.extend(TextLoader(str(path), encoding="utf-8").load())

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=100
    ).split_documents(docs)
    store = FAISS.from_documents(chunks, embeddings)
    VECTOR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(VECTOR))
    return store


def ensure_vectorstore(*, rebuild: bool = False) -> FAISS:
    if rebuild:
        if VECTOR.exists():
            shutil.rmtree(VECTOR)
        _clear_vector_cache()
        get_embeddings.cache_clear()
    return get_cached_vectorstore()


def warm_up() -> dict[str, str]:
    """Load GPU embedding model + FAISS so the first query is fast."""
    device = get_device_label()
    ensure_vectorstore()
    # Tiny encode to JIT/warm CUDA kernels
    get_embeddings().embed_query("warmup")
    return {"device": device, "embed_model": get_embed_model_name()}


def supervisor(state: TeamState) -> dict:
    hops = int(state.get("hop_count") or 0) + 1
    if hops > MAX_SUPERVISOR_HOPS:
        steps = list(state.get("steps") or [])
        steps.append(f"supervisor → finish (max hops {MAX_SUPERVISOR_HOPS} reached)")
        return {"route": "finish", "steps": steps, "hop_count": hops}

    llm = get_llm(0, max_tokens=256).with_structured_output(RouteDecision)
    decision: RouteDecision = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a supervisor for a knowledge team. "
                    "Route to researcher if evidence is missing, "
                    "analyst if notes exist but no analysis, "
                    "writer if analysis exists, finish if final_answer exists. "
                    "Keep reason under 12 words."
                )
            ),
            HumanMessage(
                content=(
                    f"Question: {state['question']}\n"
                    f"Research notes empty: {not bool(state.get('research_notes'))}\n"
                    f"Analysis empty: {not bool(state.get('analysis'))}\n"
                    f"Final empty: {not bool(state.get('final_answer'))}"
                )
            ),
        ]
    )
    steps = list(state.get("steps") or [])
    steps.append(f"supervisor → {decision.next_agent} ({decision.reason})")
    return {"route": decision.next_agent, "steps": steps, "hop_count": hops}


def researcher(state: TeamState) -> dict:
    store = ensure_vectorstore()
    docs = store.similarity_search(state["question"], k=4)
    notes = "\n\n".join(
        f"[{Path(d.metadata.get('source', 'doc')).name}]\n{d.page_content}" for d in docs
    )
    steps = list(state.get("steps") or [])
    steps.append(f"researcher retrieved {len(docs)} passage(s)")
    return {"research_notes": notes, "steps": steps}


def analyst(state: TeamState) -> dict:
    llm = get_llm(0.2, max_tokens=700)
    analysis = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Analyze retrieved notes and extract key facts + conflicts. "
                    "Be concise (bullet points preferred)."
                )
            ),
            HumanMessage(
                content=f"Question: {state['question']}\n\nNotes:\n{state['research_notes']}"
            ),
        ]
    ).content
    steps = list(state.get("steps") or [])
    steps.append("analyst synthesized findings")
    return {"analysis": analysis, "steps": steps}


def writer(state: TeamState) -> dict:
    llm = get_llm(0.3, max_tokens=700)
    answer = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Write a clear final answer for the user. "
                    "Be concise, cite document names from the analysis when possible."
                )
            ),
            HumanMessage(
                content=(
                    f"Question: {state['question']}\n\n"
                    f"Analysis:\n{state['analysis']}\n\n"
                    f"Raw notes:\n{state['research_notes']}"
                )
            ),
        ]
    ).content
    steps = list(state.get("steps") or [])
    steps.append("writer produced final answer")
    return {"final_answer": answer, "steps": steps, "route": "finish"}


def route_supervisor(state: TeamState) -> Literal["researcher", "analyst", "writer", "finish"]:
    return state.get("route", "researcher")  # type: ignore[return-value]


@lru_cache(maxsize=1)
def build_team():
    graph = StateGraph(TeamState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("researcher", researcher)
    graph.add_node("analyst", analyst)
    graph.add_node("writer", writer)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "finish": END,
        },
    )
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("writer", "supervisor")
    return graph.compile()


def _initial_state(question: str) -> TeamState:
    return {
        "question": question,
        "route": "researcher",
        "research_notes": "",
        "analysis": "",
        "final_answer": "",
        "steps": [],
        "hop_count": 0,
    }


def ask_team(question: str) -> TeamState:
    question = (question or "").strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    require_api_key()
    ensure_vectorstore()
    app = build_team()
    return app.invoke(
        _initial_state(question),
        config={"recursion_limit": MAX_SUPERVISOR_HOPS * 3},
    )


def ask_team_stream(question: str) -> Iterator[dict[str, Any]]:
    """Yield live node updates so the UI can paint progress immediately."""
    question = (question or "").strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    require_api_key()
    ensure_vectorstore()
    app = build_team()
    merged: TeamState = _initial_state(question)

    for event in app.stream(
        merged,
        config={"recursion_limit": MAX_SUPERVISOR_HOPS * 3},
        stream_mode="updates",
    ):
        for node, update in event.items():
            if isinstance(update, dict):
                for key, value in update.items():
                    merged[key] = value  # type: ignore[literal-required]
            yield {"node": node, "update": update, "state": dict(merged)}


def main() -> None:
    print("Multi-Agent Knowledge System")
    print("Supervisor → Researcher (RAG) → Analyst → Writer")
    print(f"Chat: Groq / {get_chat_model_name()}")
    print(f"Embeddings: {get_embed_model_name()} on {get_device_label()}")
    try:
        require_api_key()
        info = warm_up()
        print(f"Warmed up on {info['device']}\n")
    except ConfigurationError as exc:
        print(f"Config error: {exc}")
        return

    q = input("Question> ").strip()
    if not q:
        return
    result = ask_team(q)
    print("\n--- Trace ---")
    for step in result["steps"]:
        print(f"* {step}")
    print("\n--- Answer ---\n")
    print(result["final_answer"])


if __name__ == "__main__":
    main()
