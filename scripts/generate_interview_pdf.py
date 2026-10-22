"""Generate an interview-ready PDF brief for the Multi-Agent Knowledge System."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs" / "Aether_Knowledge_Interview_Project_Brief.pdf"
PIPELINE = BASE / "outputs" / "pipeline_overview.png"
SAMPLE = BASE / "outputs" / "sample_multi_agent_output.png"

INK = colors.HexColor("#0B1220")
TEAL = colors.HexColor("#0F766E")
MUTED = colors.HexColor("#475569")
LINE = colors.HexColor("#E2E8F0")
SOFT = colors.HexColor("#F8FAFC")
ACCENT_SOFT = colors.HexColor("#CCFBF1")


def styles():
    base = getSampleStyleSheet()
    return {
        "cover_brand": ParagraphStyle(
            "cover_brand",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=TEAL,
            alignment=TA_CENTER,
            spaceAfter=8,
            tracking=1,
        ),
        "cover_title": ParagraphStyle(
            "cover_title",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=26,
            textColor=INK,
            alignment=TA_CENTER,
            leading=32,
            spaceAfter=12,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            textColor=MUTED,
            alignment=TA_CENTER,
            leading=16,
            spaceAfter=6,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=TEAL,
            spaceBefore=14,
            spaceAfter=8,
            borderPadding=3,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=INK,
            spaceBefore=10,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            textColor=INK,
            leading=13.5,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            textColor=INK,
            leading=13,
            leftIndent=4,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            textColor=MUTED,
            leading=11,
            alignment=TA_CENTER,
        ),
        "code": ParagraphStyle(
            "code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=8,
            textColor=INK,
            leading=11,
            backColor=SOFT,
            borderPadding=6,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=10,
        ),
        "footer": ParagraphStyle(
            "footer",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
    }


def bullets(items: list[str], style) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(i, style), leftIndent=8, bulletColor=TEAL) for i in items],
        bulletType="bullet",
        start="•",
        leftIndent=12,
        spaceBefore=2,
        spaceAfter=6,
    )


def section_rule():
    data = [[""]]
    t = Table(data, colWidths=[170 * mm])
    t.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, -1), 1, LINE),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def info_table(rows: list[tuple[str, str]], col1=42 * mm, col2=128 * mm) -> Table:
    data = [
        [
            Paragraph(f"<b>{a}</b>", ParagraphStyle("c1", fontName="Helvetica", fontSize=9, textColor=MUTED)),
            Paragraph(b, ParagraphStyle("c2", fontName="Helvetica", fontSize=9, textColor=INK, leading=12)),
        ]
        for a, b in rows
    ]
    t = Table(data, colWidths=[col1, col2])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SOFT),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return t


def agent_table(s) -> Table:
    header = ["Agent", "Responsibility", "Interview talking point"]
    rows = [
        ["Supervisor", "Routes to next specialist via structured output", "Shows LangGraph conditional edges + Pydantic schemas"],
        ["Researcher", "FAISS similarity search over knowledge/", "Classic RAG retrieval, GPU/CPU embeddings"],
        ["Analyst", "Synthesizes facts and conflicts", "Separates reasoning from retrieval"],
        ["Writer", "Final user answer with citations", "Grounded generation, citation habit"],
    ]
    body_style = ParagraphStyle("td", fontName="Helvetica", fontSize=8.5, leading=11, textColor=INK)
    head_style = ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=8.5, textColor=colors.white)
    data = [[Paragraph(h, head_style) for h in header]] + [
        [Paragraph(c, body_style) for c in row] for row in rows
    ]
    t = Table(data, colWidths=[28 * mm, 62 * mm, 80 * mm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), TEAL),
                ("BACKGROUND", (0, 1), (-1, -1), SOFT),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return t


def add_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 9 * mm, "Aether Knowledge · Multi-Agent RAG · Interview Brief")
    canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build():
    s = styles()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title="Aether Knowledge — Interview Project Brief",
        author="Ali Raza",
    )
    story = []

    # Cover
    story.append(Spacer(1, 28 * mm))
    story.append(Paragraph("AETHER KNOWLEDGE", s["cover_brand"]))
    story.append(Paragraph("Multi-Agent Knowledge System", s["cover_title"]))
    story.append(
        Paragraph(
            "LangGraph · RAG · Groq · FAISS · Streamlit<br/>"
            "Interview / Portfolio Project Brief",
            s["cover_sub"],
        )
    )
    story.append(Spacer(1, 8 * mm))
    story.append(
        info_table(
            [
                ("Live demo", "https://aetherknowledge.streamlit.app/"),
                ("GitHub", "https://github.com/alirazach476/multi-agent-knowledge-system"),
                ("Status", "Working · Deployed on Streamlit Community Cloud"),
                ("License", "MIT"),
            ]
        )
    )
    story.append(Spacer(1, 10 * mm))
    story.append(
        Paragraph(
            "One-line pitch: A supervisor agent routes each question through a RAG researcher, "
            "analyst, and writer over a local knowledge base — with a live Streamlit trace.",
            s["body"],
        )
    )
    story.append(PageBreak())

    # Overview
    story.append(Paragraph("1. Project overview", s["h1"]))
    story.append(section_rule())
    story.append(
        Paragraph(
            "This project demonstrates a <b>real multi-agent loop</b> (not a fake linear script). "
            "A LangGraph supervisor inspects shared state and decides whether to call the researcher, "
            "analyst, writer, or finish. Answers are grounded in Markdown docs under <b>knowledge/</b> "
            "using FAISS vector search. Chat runs on Groq’s free API; embeddings run locally "
            "(CUDA GPU when available, CPU on Streamlit Cloud).",
            s["body"],
        )
    )
    story.append(Paragraph("Why this is interview-ready", s["h2"]))
    story.append(
        bullets(
            [
                "Shows orchestration skill (LangGraph StateGraph + conditional edges)",
                "Shows RAG fundamentals (chunking, embeddings, FAISS, citations)",
                "Shows production-ish habits (env/secrets, fail-fast config, smoke tests, deploy)",
                "Public live demo + GitHub — easy for interviewers to click and verify",
            ],
            s["bullet"],
        )
    )

    story.append(Paragraph("2. Architecture", s["h1"]))
    story.append(section_rule())
    story.append(
        Paragraph(
            "<b>User question → Supervisor loop → Researcher / Analyst / Writer → Final answer</b><br/>"
            "Specialists return control to the supervisor until <font color='#0F766E'><b>finish</b></font> "
            "(with a max hop limit to prevent infinite loops).",
            s["body"],
        )
    )
    story.append(agent_table(s))
    story.append(Spacer(1, 4 * mm))
    if PIPELINE.exists():
        img = Image(str(PIPELINE))
        img._restrictSize(170 * mm, 55 * mm)
        story.append(KeepTogether([img, Paragraph("Figure 1 — Pipeline overview (from project outputs)", s["caption"])]))

    story.append(Paragraph("3. Tech stack", s["h1"]))
    story.append(section_rule())
    story.append(
        info_table(
            [
                ("Orchestration", "LangGraph (StateGraph, conditional edges, stream updates)"),
                ("LLM / chat", "Groq API — openai/gpt-oss-20b (free tier)"),
                ("Embeddings", "sentence-transformers/all-MiniLM-L6-v2 (local; CUDA FP16 if GPU)"),
                ("Vector store", "FAISS (persisted under vectorstore/, rebuilt on demand)"),
                ("Validation", "Pydantic RouteDecision for structured supervisor routing"),
                ("UI", "Streamlit — live agent trace + branded console (Aether Knowledge)"),
                ("Config", ".env locally · Streamlit Secrets in cloud"),
                ("Tests", "unittest smoke tests (config, graph compile, corpus)"),
            ]
        )
    )

    story.append(Paragraph("4. Repository layout", s["h1"]))
    story.append(section_rule())
    story.append(
        Paragraph(
            "<font face='Courier' size='8'>"
            "app.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Streamlit UI (live trace)<br/>"
            "multi_agent.py&nbsp;&nbsp;&nbsp;&nbsp;Agents, RAG, LangGraph wiring<br/>"
            "knowledge/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5 Markdown source docs<br/>"
            "tests/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Smoke tests<br/>"
            "outputs/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Screenshots + this PDF<br/>"
            ".streamlit/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Theme + secrets example<br/>"
            "requirements.txt&nbsp;&nbsp;Pinned dependencies"
            "</font>",
            s["body"],
        )
    )
    story.append(Paragraph("Knowledge corpus (demo internal docs)", s["h2"]))
    story.append(
        bullets(
            [
                "onboarding_faq.md — API keys, models, audit logs",
                "platform_notes.md — retrieval strategy, evaluation, guardrails",
                "incident_runbook.md — SEV levels and escalation",
                "data_retention.md — transcripts, embeddings, training use",
                "support_sla.md — response targets and channels",
            ],
            s["bullet"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("5. End-to-end flow (what happens on one question)", s["h1"]))
    story.append(section_rule())
    story.append(
        bullets(
            [
                "Supervisor reads state flags (notes / analysis / answer empty?) and emits RouteDecision.",
                "Researcher embeds the query (GPU/CPU), retrieves top-k FAISS passages, writes research_notes.",
                "Analyst summarizes facts and conflicts from notes.",
                "Writer drafts a concise final answer and cites document names when possible.",
                "UI streams each node update so the interviewer sees the team collaborate live.",
            ],
            s["bullet"],
        )
    )
    if SAMPLE.exists():
        img = Image(str(SAMPLE))
        img._restrictSize(170 * mm, 95 * mm)
        story.append(KeepTogether([img, Paragraph("Figure 2 — Sample multi-agent trace and answer", s["caption"])]))

    story.append(Paragraph("6. How to run / verify (for interviewers)", s["h1"]))
    story.append(section_rule())
    story.append(Paragraph("<b>Fastest proof it works:</b> open the live demo.", s["body"]))
    story.append(
        Paragraph(
            "Live: <link href='https://aetherknowledge.streamlit.app/'>"
            "https://aetherknowledge.streamlit.app/</link>",
            s["body"],
        )
    )
    story.append(Paragraph("Local run", s["h2"]))
    story.append(
        Paragraph(
            "<font face='Courier' size='8'>"
            "python -m venv .venv<br/>"
            ".venv\\Scripts\\activate&nbsp;&nbsp;# Windows<br/>"
            "pip install -r requirements.txt<br/>"
            "copy .env.example .env&nbsp;&nbsp;# set GROQ_API_KEY<br/>"
            "streamlit run app.py<br/>"
            "python -m unittest discover -s tests -v"
            "</font>",
            s["body"],
        )
    )
    story.append(Paragraph("Example questions to demo", s["h2"]))
    story.append(
        bullets(
            [
                "How do I get API access and how often do keys expire?",
                "What retrieval strategy and chunk sizes do we use?",
                "Where are audit logs and how long are they retained?",
                "What is the on-call escalation path for a production RAG outage?",
            ],
            s["bullet"],
        )
    )

    story.append(Paragraph("7. Design decisions & trade-offs (say this in interviews)", s["h1"]))
    story.append(section_rule())
    story.append(
        bullets(
            [
                "<b>Supervisor loop vs linear chain:</b> looping mirrors real agent teams and can revisit research if needed.",
                "<b>Groq instead of paid OpenAI:</b> free chat API; embeddings stay local so RAG still works offline from LLM vendor.",
                "<b>Structured routing (Pydantic):</b> reduces free-text routing errors vs parsing natural language.",
                "<b>Hop / recursion limits:</b> guards against infinite supervisor cycles.",
                "<b>GPU optional:</b> CUDA FP16 locally for speed; auto CPU on Streamlit Cloud so deploy stays portable.",
                "<b>Small corpus by design:</b> clear demo of grounding — not a claim of enterprise search scale.",
            ],
            s["bullet"],
        )
    )

    story.append(Paragraph("8. What I would improve next", s["h1"]))
    story.append(section_rule())
    story.append(
        bullets(
            [
                "Hybrid retrieval (keyword + dense) and metadata filters",
                "Evaluation harness (faithfulness / relevancy on a golden set)",
                "Streaming token UI for the writer agent",
                "Auth + per-tenant knowledge bases for a production SaaS shape",
                "CI: lint + smoke tests on every push",
            ],
            s["bullet"],
        )
    )

    story.append(Spacer(1, 8 * mm))
    story.append(section_rule())
    story.append(
        Paragraph(
            "Prepared as an interview portfolio brief · Project is live and working · "
            "Source: github.com/alirazach476/multi-agent-knowledge-system",
            s["small"],
        )
    )

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
