"""Streamlit UI for the multi-agent knowledge system."""

import streamlit as st

from multi_agent import ask_team

st.set_page_config(page_title="Multi-Agent Knowledge", page_icon="🤝", layout="wide")
st.title("Multi-Agent Knowledge System")
st.caption("Supervisor routes between RAG Researcher → Analyst → Writer")

q = st.text_input("Ask the agent team")
if st.button("Run multi-agent workflow", type="primary") and q.strip():
    with st.spinner("Agents collaborating..."):
        result = ask_team(q.strip())
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Trace")
        for step in result["steps"]:
            st.write(f"- {step}")
    with right:
        st.subheader("Final answer")
        st.write(result["final_answer"])
        with st.expander("Research notes"):
            st.write(result["research_notes"])
        with st.expander("Analysis"):
            st.write(result["analysis"])
