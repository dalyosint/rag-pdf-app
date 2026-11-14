import pydantic

# These models describe what each Inngest step returns.
# Inngest saves every step's output; with PydanticSerializer it can turn that
# saved data back into these typed objects (and validates it on the way).


class RAGChunkAndSrc(pydantic.BaseModel):
    """Output of the "load-and-chunk" step."""
    chunks: list[str]
    source_id: str  # name of the PDF the chunks came from


class RAGUpsertResult(pydantic.BaseModel):
    """Output of the "embed-and-upsert" step."""
    ingested: int  # how many chunks were stored


class RAGSearchResult(pydantic.BaseModel):
    """Output of the "embed-and-search" step."""
    contexts: list[str]  # text of the most relevant chunks
    sources: list[str]   # PDFs those chunks came from


class RAGQueryResult(pydantic.BaseModel):
    """Final answer returned by the query function."""
    answer: str
    sources: list[str]
    num_contexts: int
    contexts: list[str] = []  # the chunks used, so the UI can show them
