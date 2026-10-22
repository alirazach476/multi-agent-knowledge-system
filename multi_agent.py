"""
Multi-Agent Knowledge System

Supervisor routes to specialized agents:
- Researcher (RAG over docs)
- Analyst (structured synthesis)
- Writer (final answer)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

BASE = Path(__file__).resolve().parent
DOCS = BASE / "knowledge"
VECTOR = BASE / "vectorstore"


class RouteDecision(BaseModel):
    next_agent: Literal["researcher", "analyst", "writer", "finish"]
    reason: str


class TeamState(TypedDict):
    question: str
    route: str
    research_notes: str
    analysis: str
    final_answer: str
    steps: list[str]


def get_llm(temp: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=temp)


def ensure_vectorstore() -> FAISS:
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    if (VECTOR / "index.faiss").exists():
        return FAISS.load_local(str(VECTOR), embeddings, allow_dangerous_deserialization=True)

    DOCS.mkdir(parents=True, exist_ok=True)
    docs = []
    for path in DOCS.glob("**/*"):
        if path.suffix.lower() in {".txt", ".md"}:
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
    if not docs:
        raise FileNotFoundError(f"Add documents under {DOCS}")

    chunks = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100).split_documents(docs)
    store = FAISS.from_documents(chunks, embeddings)
    VECTOR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(VECTOR))
    return store


def supervisor(state: TeamState) -> dict:
    llm = get_llm(0).with_structured_output(RouteDecision)
    decision: RouteDecision = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a supervisor for a knowledge team. "
                    "Route to researcher if evidence is missing, "
                    "analyst if notes exist but no analysis, "
                    "writer if analysis exists, finish if final_answer exists."
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
    return {"route": decision.next_agent, "steps": steps}


def researcher(state: TeamState) -> dict:
    store = ensure_vectorstore()
    docs = store.similarity_search(state["question"], k=4)
    notes = "\n\n".join(
        f"[{Path(d.metadata.get('source', 'doc')).name}]\n{d.page_content}" for d in docs
    )
    steps = list(state.get("steps") or [])
    steps.append("researcher retrieved context")
    return {"research_notes": notes, "steps": steps}


def analyst(state: TeamState) -> dict:
    llm = get_llm(0.2)
    analysis = llm.invoke(
        [
            SystemMessage(content="Analyze retrieved notes and extract key facts + conflicts."),
            HumanMessage(
                content=f"Question: {state['question']}\n\nNotes:\n{state['research_notes']}"
            ),
        ]
    ).content
    steps = list(state.get("steps") or [])
    steps.append("analyst synthesized findings")
    return {"analysis": analysis, "steps": steps}


def writer(state: TeamState) -> dict:
    llm = get_llm(0.3)
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


def ask_team(question: str) -> TeamState:
    app = build_team()
    return app.invoke(
        {
            "question": question,
            "route": "researcher",
            "research_notes": "",
            "analysis": "",
            "final_answer": "",
            "steps": [],
        }
    )


def main() -> None:
    print("Multi-Agent Knowledge System (supervisor + RAG researcher + analyst + writer)")
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
