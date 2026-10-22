# Multi-Agent Knowledge System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-aetherknowledge.streamlit.app-FF4B4B?logo=streamlit&logoColor=white)](https://aetherknowledge.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Live app:** [https://aetherknowledge.streamlit.app/](https://aetherknowledge.streamlit.app/)

A **LangGraph multi-agent RAG** demo: a supervisor routes each question through specialist agents that retrieve, analyze, and write grounded answers over a local Markdown knowledge base.

```
User question
     │
     ▼
┌────────────┐
│ Supervisor │ ←──── loop until finish (max hops)
└─────┬──────┘
      │
      ├─► Researcher  (FAISS similarity search)
      ├─► Analyst     (synthesize facts / conflicts)
      └─► Writer      (final answer + citations)
```

| Agent | Responsibility |
|-------|----------------|
| **Supervisor** | Structured routing (`researcher` / `analyst` / `writer` / `finish`) |
| **Researcher** | RAG retrieval over `knowledge/` via FAISS |
| **Analyst** | Synthesizes evidence and flags conflicts |
| **Writer** | Produces the user-facing answer with document citations |

## Features

- Real supervisor **loop** (not a one-shot chain)
- Persistent local FAISS index under `vectorstore/` (gitignored)
- CLI + Streamlit UI sharing the same `ask_team()` entry point
- Free stack: **Groq** chat API + **local GPU** MiniLM embeddings (CUDA when available)
- Live Streamlit trace that paints each agent step as it finishes
- Fail-fast config checks for missing API keys and empty corpora
- Sample internal-docs corpus (onboarding, platform, incidents, retention, SLA)

## Requirements

- Python **3.10+**
- A free [Groq API key](https://console.groq.com/keys) (chat only)
- Embeddings run **locally** via `sentence-transformers` (no paid key)

## Setup

```bash
cd multi-agent-knowledge-system
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Copy the env template and set your free Groq key:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `GROQ_API_KEY` | — | Required (free from Groq Console) |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Chat model for agents |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |

## Run

```bash
# CLI
python multi_agent.py

# Streamlit UI
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (already: `alirazach476/multi-agent-knowledge-system`).
2. Open [share.streamlit.io](https://share.streamlit.io/) → **New app**.
3. Select the repo, branch `main`, main file **`app.py`**.
4. Under **Advanced settings → Secrets**, paste:

```toml
GROQ_API_KEY = "gsk_your-real-key"
GROQ_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DEVICE = "cpu"
```

5. Click **Deploy**. First build can take several minutes (downloads `torch` + MiniLM).

**Notes**
- Streamlit Cloud has **no GPU** — the app auto-uses CPU embeddings (still free).
- Never commit `.env` or `.streamlit/secrets.toml` (both are gitignored).
- If the app is killed for memory, reboot from the Cloud dashboard; MiniLM is intentionally small.

### Example questions

- How do I get API access and how often do keys expire?
- What retrieval strategy and chunk sizes do we use?
- Where are audit logs and how long are they retained?
- What is the on-call escalation path for a production RAG outage?

## Project layout

```
├── app.py              # Streamlit UI
├── multi_agent.py      # Agents, RAG, LangGraph wiring
├── knowledge/          # Source documents for retrieval
├── outputs/            # Sample screenshots
├── tests/              # Smoke tests (no API key required)
├── .env.example
├── requirements.txt
└── LICENSE
```

## How it works

1. **Supervisor** inspects state (notes / analysis / answer) and picks the next agent via a Pydantic-structured decision.
2. **Researcher** loads or builds a FAISS index from `knowledge/**/*.md`, then retrieves top passages.
3. **Analyst** and **Writer** call the chat model with that context.
4. Control returns to the supervisor until `finish` (or a hop limit is hit).

> **Note:** Loading the local FAISS store uses `allow_dangerous_deserialization=True` because LangChain persists the index with pickle. Only load indexes you built yourself.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `GROQ_API_KEY is missing` | Create a free key at [console.groq.com/keys](https://console.groq.com/keys) and put it in `.env` |
| `No knowledge documents found` | Add `.md` / `.txt` files under `knowledge/` |
| Stale / dimension errors after switching from OpenAI | Delete `vectorstore/` or use **Rebuild vector index** (embedding model changed) |
| Groq rate limits / model 404 | Wait a moment, or set `GROQ_MODEL=openai/gpt-oss-120b` |

## Tests

```bash
python -m unittest discover -s tests -v
```

## Sample outputs

![Pipeline](outputs/pipeline_overview.png)
![Multi-Agent Output](outputs/sample_multi_agent_output.png)

## Stack

LangChain · LangGraph · FAISS · Groq (free chat) · local HuggingFace embeddings · Streamlit · Pydantic

## License

MIT — see [LICENSE](LICENSE).
