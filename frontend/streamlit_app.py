"""Minimal Streamlit chat UI that calls the deployed /chat endpoint."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("RAGBedRockChatbot")
st.caption("Ask a question about the knowledge base.")

if question := st.chat_input("Your question"):
    st.chat_message("user").write(question)
    resp = requests.post(f"{API_URL}/chat", json={"question": question}, timeout=60)
    data = resp.json()
    with st.chat_message("assistant"):
        st.write(data["answer"])
        if data.get("citations"):
            st.caption("Sources: " + ", ".join(data["citations"]))
