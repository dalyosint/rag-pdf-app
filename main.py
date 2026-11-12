import logging

import inngest
import inngest.fast_api
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

# The Inngest client: it registers our functions with the Inngest server.
# is_production=False means "talk to the local dev server" (no signing key needed).
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


# A first function to check that everything is wired up.
# It runs whenever an event named "rag/ingest_pdf" is sent.
@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    return {"hello": "world"}


app = FastAPI()

# Exposes the functions at /api/inngest so the Inngest dev server can find them.
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf])
