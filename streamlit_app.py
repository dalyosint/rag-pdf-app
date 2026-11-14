import os
import time
from pathlib import Path

import inngest
import requests
import streamlit as st
from dotenv import load_dotenv

from vector_db import QdrantStorage

load_dotenv()

st.set_page_config(page_title="PDF Chat", page_icon="📚", layout="centered")

INNGEST_API = os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")
UPLOAD_DIR = Path("uploads")
SUGGESTIONS = [
    "Summarize the main points",
    "What are the key definitions?",
    "List the most important numbers or dates",
]


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


def send_event(name: str, data: dict) -> str:
    """Send an event to Inngest and return its id."""
    ids = get_inngest_client().send_sync(inngest.Event(name=name, data=data))
    return ids[0]


def wait_for_run(event_id: str, timeout_s: float = 120, no_run_after_s: float | None = None) -> dict | None:
    """Poll the Inngest dev server until the run started by this event finishes.

    Returns the run's output, or None if no run was created within
    `no_run_after_s` seconds (Inngest skipped it, e.g. because of the rate limit).
    """
    start = time.time()
    while True:
        resp = requests.get(f"{INNGEST_API}/events/{event_id}/runs", timeout=5)
        resp.raise_for_status()
        runs = resp.json().get("data", [])
        elapsed = time.time() - start
        if runs:
            status = runs[0].get("status")
            if status == "Completed":
                return runs[0].get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"The run {status.lower()}. Check the Inngest dashboard for details.")
        elif no_run_after_s is not None and elapsed > no_run_after_s:
            return None
        if elapsed > timeout_s:
            raise TimeoutError("Timed out waiting for the answer. Check the Inngest dashboard.")
        time.sleep(0.5)


def list_documents() -> dict[str, int] | None:
    """{pdf name: chunk count} from Qdrant, or None if Qdrant is unreachable."""
    try:
        return QdrantStorage().list_sources()
    except Exception:
        return None


def save_upload(file) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / file.name
    path.write_bytes(file.getbuffer())
    return path


def escape_md(text: str) -> str:
    # "$" would otherwise be rendered as LaTeX math by Streamlit
    return text.replace("$", "\\$")


# ---------------------------------------------------------------------------
# UI pieces
# ---------------------------------------------------------------------------
def render_sources(sources: list[str], contexts: list[str]) -> None:
    if not sources:
        return
    label = f"Sources · {len(sources)} document{'s' if len(sources) != 1 else ''}, {len(contexts)} passages"
    with st.expander(label):
        st.markdown(" ".join(f"`📄 {s}`" for s in sources))
        for i, ctx in enumerate(contexts, 1):
            with st.container(border=True):
                st.caption(f"Passage {i}")
                snippet = ctx if len(ctx) <= 600 else ctx[:600] + "…"
                st.markdown(escape_md(snippet))


def ingest_files(files) -> None:
    for file in files:
        with st.status(f"Adding **{file.name}**…", expanded=False) as status:
            try:
                path = save_upload(file)
                status.write("Saved file, starting ingestion…")
                event_id = send_event(
                    "rag/ingest_pdf",
                    {"pdf_path": str(path.resolve()), "source_id": path.name},
                )
                status.write("Reading, chunking and embedding…")
                out = wait_for_run(event_id, timeout_s=300, no_run_after_s=15)
                if out is None:
                    status.update(label=f"**{file.name}** was already added recently (skipped)", state="complete")
                else:
                    status.update(label=f"**{file.name}** · {out.get('ingested', 0)} chunks added", state="complete")
                    st.toast(f"{file.name} is ready", icon="✅")
            except requests.ConnectionError:
                status.update(label=f"**{file.name}**: can't reach the Inngest server", state="error")
            except Exception as e:
                status.update(label=f"**{file.name}**: {e}", state="error")


def answer_question(question: str, top_k: int) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(escape_md(question))

    with st.chat_message("assistant"):
        try:
            with st.status("Searching your documents…") as status:
                event_id = send_event("rag/query_pdf_ai", {"question": question, "top_k": top_k})
                out = wait_for_run(event_id)
                status.update(label="Done", state="complete", expanded=False)
        except requests.ConnectionError:
            st.error(f"Can't reach the Inngest server at {INNGEST_API}. Is it running?")
            return
        except Exception as e:
            st.error(str(e))
            return

        answer = out.get("answer") or "_No answer returned._"
        sources, contexts = out.get("sources", []), out.get("contexts", [])
        st.markdown(escape_md(answer))
        render_sources(sources, contexts)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "contexts": contexts}
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----- Sidebar: documents + settings -----
with st.sidebar:
    st.title("📚 PDF Chat")
    st.caption("Chat with your documents")

    st.subheader("Add documents")
    with st.form("upload", clear_on_submit=True, border=False):
        files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
        submitted = st.form_submit_button("Add to library", type="primary", use_container_width=True)
    if submitted and files:
        ingest_files(files)

    st.subheader("Library")
    docs = list_documents()
    if docs is None:
        st.error("Can't reach Qdrant on localhost:6333.")
        docs = {}
    elif not docs:
        st.caption("No documents yet.")
    else:
        for name, n in sorted(docs.items()):
            st.markdown(f"📄 **{name}**  \n<small>{n} chunks</small>", unsafe_allow_html=True)

    st.subheader("Settings")
    top_k = st.slider("Passages to retrieve", 1, 10, 5, help="How many chunks are sent to the LLM as context.")
    if st.button("Clear chat", use_container_width=True, disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()

# ----- Main area: chat -----
st.title("Ask your PDFs")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(escape_md(msg["content"]))
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []), msg.get("contexts", []))

pending = st.session_state.pop("pending", None)
if not st.session_state.messages and not pending:
    if not docs:
        st.info("👈 Add a PDF in the sidebar to get started.")
    else:
        st.caption("Try one of these:")
        cols = st.columns(len(SUGGESTIONS))
        for col, suggestion in zip(cols, SUGGESTIONS):
            if col.button(suggestion, use_container_width=True):
                st.session_state.pending = suggestion
                st.rerun()

prompt = st.chat_input("Ask a question about your PDFs…", disabled=not docs)
question = prompt or pending
if question and question.strip():
    answer_question(question.strip(), top_k)
