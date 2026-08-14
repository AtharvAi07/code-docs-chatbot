"""
Streamlit UI for the Code Documentation Chatbot.
Custom-styled dark theme - sharp corners, monospace accents, minimal motion.
"""

import streamlit as st
import time

from github_fetcher import GitHubFetcher
from chunker import SmartChunker
from embeddings import EmbeddingGenerator
from vector_db import VectorDB
from rag_chain import RAGChain


st.set_page_config(
    page_title="repo-chat",
    page_icon=":black_small_square:",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    @import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap");

    :root {
        --bg: #1a1d23;
        --bg-alt: #202329;
        --border: #2e323b;
        --text: #e8e6e1;
        --text-dim: #8a8d96;
        --accent: #5b8dbe;
        --accent-dim: #3d5f80;
    }

    * {
        font-family: "Inter", sans-serif;
    }

    code, .mono {
        font-family: "IBM Plex Mono", monospace !important;
    }

    #MainMenu, header, footer { visibility: hidden; }
    .stDeployButton { display: none; }

    .stApp {
        background-color: var(--bg);
        color: var(--text);
    }

    section[data-testid="stSidebar"] {
        background-color: var(--bg-alt);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    .stButton button, .stTextInput input, .stTextArea textarea {
        border-radius: 2px !important;
        background-color: var(--bg) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        box-shadow: none !important;
    }
    .stButton button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }
    .stButton button:focus {
        box-shadow: 0 0 0 1px var(--accent) !important;
    }

    h1, h2, h3 {
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }

    hr {
        border-color: var(--border) !important;
        margin: 1.2rem 0 !important;
    }

    .repo-item {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.82rem;
        padding: 6px 10px;
        border: 1px solid var(--border);
        margin-bottom: 6px;
        color: var(--text-dim);
        display: flex;
        justify-content: space-between;
        animation: fadeIn 0.3s ease;
    }
    .repo-item .• {
        color: var(--accent);
        margin-right: 6px;
    }

    .status-line {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.78rem;
        color: var(--accent-dim);
        padding: 4px 0;
        animation: pulse 1.4s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .msg {
        border: 1px solid var(--border);
        padding: 14px 16px;
        margin-bottom: 12px;
        animation: fadeIn 0.35s ease;
    }
    .msg-user {
        background-color: var(--bg-alt);
        border-left: 2px solid var(--accent);
    }
    .msg-bot {
        background-color: var(--bg);
        border-left: 2px solid var(--text-dim);
    }
    .msg-label {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-dim);
        margin-bottom: 6px;
    }
    .msg-sources {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.72rem;
        color: var(--accent-dim);
        margin-top: 8px;
        border-top: 1px solid var(--border);
        padding-top: 8px;
    }

    .empty-state {
        text-align: center;
        padding: 60px 20px;
        color: var(--text-dim);
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


if "vector_db" not in st.session_state:
    st.session_state.vector_db = VectorDB(dimension=384)
if "embedder" not in st.session_state:
    with st.spinner(""):
        st.session_state.embedder = EmbeddingGenerator()
if "chunker" not in st.session_state:
    st.session_state.chunker = SmartChunker()
if "repos_loaded" not in st.session_state:
    st.session_state.repos_loaded = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag" not in st.session_state:
    st.session_state.rag = None


with st.sidebar:
    st.markdown("### repo-chat")
    st.markdown(
        "<span class=mono style=font-size:0.75rem;color:var(--text-dim);>"
        "query github repos in natural language</span>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown(
        "<div class=mono style=font-size:0.72rem;color:var(--text-dim);"
        "text-transform:uppercase;letter-spacing:0.08em;>add repository</div>",
        unsafe_allow_html=True
    )

    repo_input = st.text_input(
        "repo",
        placeholder="owner/repo",
        label_visibility="collapsed"
    )

    fetch_clicked = st.button("index repository", use_container_width=True)

    if fetch_clicked and repo_input:
        if "/" not in repo_input:
            st.error("format: owner/repo")
        else:
            owner, repo = repo_input.strip().split("/", 1)
            status = st.empty()

            status.markdown(
                f"<div class=status-line>fetching {owner}/{repo}...</div>",
                unsafe_allow_html=True
            )
            fetcher = GitHubFetcher()
            files = fetcher.fetch_repo(owner, repo, max_files=20)

            if not files:
                status.error(f"could not fetch {owner}/{repo}")
            else:
                status.markdown(
                    f"<div class=status-line>chunking {len(files)} files...</div>",
                    unsafe_allow_html=True
                )
                all_chunks = []
                for f in files:
                    file_chunks = st.session_state.chunker.chunk_file(
                        f["content"], f["name"], f["type"]
                    )
                    for c in file_chunks:
                        c["source_url"] = f.get("url", "")
                        c["repo"] = f.get("repo", f"{owner}/{repo}")
                    all_chunks.extend(file_chunks)

                status.markdown(
                    f"<div class=status-line>embedding {len(all_chunks)} chunks...</div>",
                    unsafe_allow_html=True
                )
                all_chunks = st.session_state.embedder.embed_chunks(all_chunks)
                st.session_state.vector_db.add_chunks(all_chunks)

                st.session_state.repos_loaded.append({
                    "name": f"{owner}/{repo}",
                    "files": len(files),
                    "chunks": len(all_chunks)
                })

                st.session_state.rag = RAGChain(
                    vector_db=st.session_state.vector_db,
                    embedding_generator=st.session_state.embedder
                )

                status.empty()
                st.rerun()

    st.markdown("---")
    st.markdown(
        "<div class=mono style=font-size:0.72rem;color:var(--text-dim);"
        "text-transform:uppercase;letter-spacing:0.08em;>indexed repos</div>",
        unsafe_allow_html=True
    )

    if not st.session_state.repos_loaded:
        st.markdown(
            "<div style=font-size:0.78rem;color:var(--text-dim);margin-top:8px;>"
            "none yet</div>",
            unsafe_allow_html=True
        )
    else:
        for r in st.session_state.repos_loaded:
            st.markdown(
                f"<div class=repo-item>"
                f"<span><span class=•>•</span>{r['name']}</span>"
                f"<span>{r['chunks']} chunks</span>"
                f"</div>",
                unsafe_allow_html=True
            )


st.markdown("## Ask your codebase")

if not st.session_state.repos_loaded:
    st.markdown(
        "<div class=empty-state>"
        "no repositories indexed yet<br>"
        "add one from the sidebar to start asking questions"
        "</div>",
        unsafe_allow_html=True
    )
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"<div class=\"msg msg-user\">"
                f"<div class=msg-label>you</div>{msg['content']}"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            sources_html = ""
            if msg.get("sources"):
                sources_str = " | ".join(msg["sources"])
                sources_html = f"<div class=msg-sources>sources: {sources_str}</div>"

            st.markdown(
                f"<div class=\"msg msg-bot\">"
                f"<div class=msg-label>repo-chat</div>{msg['content']}"
                f"{sources_html}"
                f"</div>",
                unsafe_allow_html=True
            )

    query = st.chat_input("ask a question about the indexed repos...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        status_placeholder = st.empty()
        status_placeholder.markdown(
            "<div class=status-line>searching relevant chunks...</div>",
            unsafe_allow_html=True
        )

        result = st.session_state.rag.answer(query, top_k=5)

        status_placeholder.empty()

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })

        st.rerun()
