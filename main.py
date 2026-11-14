import datetime
import logging
import os
import uuid

import inngest
import inngest.fast_api
from dotenv import load_dotenv
from fastapi import FastAPI
from inngest.experimental import ai

from custom_types import RAGChunkAndSrc, RAGQueryResult, RAGSearchResult, RAGUpsertResult
from data_loader import embed_texts, load_and_chunk_pdf
from vector_db import QdrantStorage

load_dotenv()

# The Inngest client: it registers our functions with the Inngest server.
# is_production=False means "talk to the local dev server" (no signing key needed).
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


# ---------------------------------------------------------------------------
# 1) Ingest a PDF:  PDF -> chunks -> embeddings -> Qdrant
# ---------------------------------------------------------------------------
@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    # At most 2 ingestions start per minute (extra ones wait in a queue)
    throttle=inngest.Throttle(limit=2, period=datetime.timedelta(minutes=1)),
    # The same PDF (source_id) is ingested at most once every 4 hours
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
    ),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load() -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        # Same PDF + same chunk number -> same id, so re-ingesting overwrites
        # the old points instead of creating duplicates.
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunk} for chunk in chunks]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    # Each step's result is saved by Inngest. If a later step fails,
    # the retry reuses saved results instead of redoing finished steps.
    chunks_and_src = await ctx.step.run("load-and-chunk", _load, output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run(
        "embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult
    )
    return ingested.model_dump()


# ---------------------------------------------------------------------------
# 2) Ask a question:  question -> embedding -> Qdrant search -> LLM answer
# ---------------------------------------------------------------------------
@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        found = QdrantStorage().search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run(
        "embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult
    )

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )

    # step.ai.infer: the Inngest server makes the OpenAI call itself,
    # and the prompt + response show up in the Inngest dashboard.
    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content},
            ],
        },
    )

    answer = res["choices"][0]["message"]["content"].strip()
    return RAGQueryResult(
        answer=answer, sources=found.sources, num_contexts=len(found.contexts)
    ).model_dump()


app = FastAPI()

# Exposes both functions at /api/inngest so the Inngest dev server can find them.
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
