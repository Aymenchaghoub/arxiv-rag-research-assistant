"""Streamlit UI (Phase 7)."""

from __future__ import annotations

import asyncio
import os

import httpx
import streamlit as st

__all__ = ["main"]

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8001")


def inject_custom_css():
    """Injects custom CSS based on ui-ux-pro-max academic/minimal recommendations."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Crimson+Pro:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Atkinson Hyperlegible', sans-serif !important;
            color: #171717 !important;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Crimson Pro', serif !important;
            color: #171717 !important;
            font-weight: 600 !important;
        }

        .stApp {
            background-color: #FFFFFF !important;
        }

        [data-testid="stSidebar"] {
            background-color: #F8F9FA !important;
            border-right: 1px solid #E5E7EB !important;
        }

        .stButton > button {
            background-color: #D4AF37 !important;
            color: #171717 !important;
            border: none !important;
            border-radius: 4px !important;
            font-weight: 600 !important;
            transition: opacity 0.2s ease-in-out !important;
            cursor: pointer !important;
        }
        .stButton > button:hover {
            opacity: 0.85 !important;
            color: #171717 !important;
        }

        .stMarkdown p {
            color: #404040 !important;
            font-size: 1.05rem;
            line-height: 1.6;
        }

        [data-testid="stChatMessage"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E5E7EB !important;
            border-radius: 6px !important;
            padding: 1rem !important;
            box-shadow: none !important;
        }

        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            border: 1px solid #D1D5DB !important;
            border-radius: 4px !important;
        }
        
        div[data-baseweb="input"] > div:focus-within,
        div[data-baseweb="select"] > div:focus-within {
            border-color: #D4AF37 !important;
            box-shadow: 0 0 0 1px #D4AF37 !important;
        }
        
        a {
            color: #D4AF37 !important;
            text-decoration: none !important;
        }
        a:hover {
            text-decoration: underline !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


async def ingest_document(arxiv_id: str, chunking_strategy: str, chunk_size: int) -> int:
    async with httpx.AsyncClient() as client:
        payload = {
            "arxiv_id": arxiv_id,
            "chunking_strategy": chunking_strategy,
            "chunk_size": chunk_size,
        }
        resp = await client.post(f"{API_BASE_URL}/api/v1/ingest", json=payload, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        return data["document_count"]


async def chat_query(question: str, collection_name: str, k: int) -> dict:
    async with httpx.AsyncClient() as client:
        payload = {
            "question": question,
            "collection_name": collection_name,
            "retriever_type": "dense",
            "k": k,
        }
        resp = await client.post(f"{API_BASE_URL}/api/v1/chat", json=payload, timeout=60.0)
        resp.raise_for_status()
        return resp.json()


def render_sidebar():
    with st.sidebar:
        st.markdown("### Configuration")
        hf_token = st.text_input("HuggingFace API Token", type="password", placeholder="hf_...")
        if hf_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

        st.markdown("---")
        st.markdown("### Document Ingestion")
        arxiv_id = st.text_input("ArXiv ID", placeholder="e.g. 1706.03762")
        strategy = st.selectbox("Chunking Strategy", ["fixed_size", "sentence"])

        # We need chunk_size. Standard fallback for fixed_size.
        chunk_size = 512
        if strategy == "fixed_size":
            chunk_size = st.number_input("Chunk Size", value=512, step=100)

        k_val = st.slider("K (Retrieval docs)", min_value=1, max_value=10, value=5)
        st.session_state["k_val"] = k_val

        if st.button("Process Document"):
            if not arxiv_id:
                st.error("Please enter an ArXiv ID.")
            else:
                with st.spinner(f"Ingesting {arxiv_id}..."):
                    try:
                        # Fallback for Strategy names to match chunkers if needed
                        chunk_strat = "token" if strategy == "fixed_size" else strategy
                        count = asyncio.run(ingest_document(arxiv_id, chunk_strat, int(chunk_size)))
                        st.success(f"Successfully processed! Indexed {count} chunks.")
                        # Format collection name
                        collection_name = arxiv_id.replace(".", "-").replace("/", "-")
                        if not collection_name[0].isalpha():
                            collection_name = f"arxiv-{collection_name}"
                        st.session_state["active_collection"] = collection_name
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")

        st.markdown("---")
        st.markdown("### Knowledge Base")
        active = st.session_state.get("active_collection")
        if active:
            st.info(f"Active Collection: {active}")
        else:
            st.info("No documents indexed yet.")


def render_chat_tab():
    col1, col2 = st.columns([2, 1])

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am your ArXiv Research Assistant. "
                    "Please index a document in the sidebar to begin."
                ),
            }
        ]
    if "current_sources" not in st.session_state:
        st.session_state["current_sources"] = []

    with col1:
        st.markdown("### Research Chat")

        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).markdown(msg["content"])

        query = st.chat_input("Ask a question about the document...")
        if query:
            if not st.session_state.get("active_collection"):
                st.warning("Please process a document first.")
            else:
                st.session_state["messages"].append({"role": "user", "content": query})
                st.chat_message("user").markdown(query)

                with st.spinner("Thinking..."):
                    try:
                        resp = asyncio.run(
                            chat_query(
                                question=query,
                                collection_name=st.session_state["active_collection"],
                                k=st.session_state.get("k_val", 5),
                            )
                        )
                        ans = resp["answer"]
                        sources = resp["sources"]

                        st.session_state["messages"].append({"role": "assistant", "content": ans})
                        st.session_state["current_sources"] = sources
                        st.chat_message("assistant").markdown(ans)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Query failed: {str(e)}")

    with col2:
        st.markdown("### Context Sources")
        with st.expander("Retrieved Sources", expanded=True):
            sources = st.session_state.get("current_sources", [])
            if not sources:
                st.markdown("*Context will appear here during chat.*")
            else:
                for idx, src in enumerate(sources, 1):
                    arxiv = src.get("arxiv_id", "Unknown")
                    page = src.get("page", "N/A")
                    txt = src.get("page_content", "")[:300] + "..."
                    st.markdown(f"**Source {idx}** (ArXiv: {arxiv}, Page: {page})")
                    st.caption(f"{txt}")
                    st.divider()


def render_evaluation_tab():
    st.markdown("### RAGAS Evaluation Pipeline")
    st.markdown("Run programmatic evaluation using `ragas` against the active collection.")

    if not st.session_state.get("active_collection"):
        st.warning("Please ingest a document first.")
        return

    questions = st.text_area(
        "Test Questions (one per line)", "What is the main contribution of this paper?"
    )
    truths = st.text_area(
        "Ground Truth Answers (one per line)", "The main contribution is a novel RAG pipeline."
    )

    if st.button("Run Evaluation"):
        q_list = [q.strip() for q in questions.split("\n") if q.strip()]
        t_list = [t.strip() for t in truths.split("\n") if t.strip()]

        if len(q_list) != len(t_list):
            st.error("Number of questions must match number of ground truths.")
            return

        with st.spinner("Running deep evaluation... (This may take a while)"):
            try:
                from app.evaluation.ragas_evaluator import RAGASEvaluator

                evaluator = RAGASEvaluator()
                df = evaluator.evaluate_pipeline(
                    questions=q_list,
                    ground_truths=t_list,
                    collection_name=st.session_state["active_collection"],
                )

                st.success("Evaluation complete!")
                st.dataframe(df)

                numeric_cols = df.select_dtypes(include="number").columns
                if len(numeric_cols) > 0:
                    avgs = df[numeric_cols].mean().to_frame("Average Score").reset_index()
                    st.bar_chart(data=avgs, x="index", y="Average Score")
            except Exception as e:
                st.error(f"Evaluation failed: {e}")


def main() -> None:
    st.set_page_config(
        page_title="ArXiv RAG Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    st.markdown("<h1>ArXiv RAG Research Assistant</h1>", unsafe_allow_html=True)
    st.markdown(
        "An AI-powered academic reading assistant. "
        "Ingest papers, chat with their content, and retrieve citations with ease."
    )
    st.markdown("---")

    render_sidebar()

    tab1, tab2 = st.tabs(["Research Chat", "Evaluation"])
    with tab1:
        render_chat_tab()
    with tab2:
        render_evaluation_tab()


if __name__ == "__main__":
    main()
