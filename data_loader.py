from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from openai import OpenAI

load_dotenv()

# Reads OPENAI_API_KEY from the environment (.env)
client = OpenAI()

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072  # must match the vector size of the Qdrant collection

# Splits text into chunks of ~1000 tokens. Neighbouring chunks share 200 tokens,
# so a sentence cut at a chunk border still appears whole in one of them.
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_pdf(path: str) -> list[str]:
    """Read a PDF and return its text as a list of chunks."""
    docs = PDFReader().load_data(file=path)  # one document per page
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Turn each text into a vector of EMBED_DIM numbers using OpenAI."""
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]
