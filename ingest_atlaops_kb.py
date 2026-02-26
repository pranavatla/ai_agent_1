import argparse
from pathlib import Path
import uuid

import chromadb
from chromadb.utils import embedding_functions


KB_DIR = Path("docs/atlaops-kb")
CHROMA_PATH = Path("chroma_db")
COLLECTION_NAME = "atlaops_kb"


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not normalized:
        return []

    chunks = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = max(0, end - overlap)
    return chunks


def list_kb_files() -> list[Path]:
    if not KB_DIR.exists():
        return []
    files = []
    for path in KB_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            files.append(path)
    return sorted(files)


def get_embedding_fn(api_key: str):
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
    )


def ingest(reset: bool, api_key: str):
    files = list_kb_files()
    if not files:
        raise RuntimeError(f"No .md/.txt files found under {KB_DIR}")

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    embedding_fn = get_embedding_fn(api_key)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    if reset:
        client.delete_collection(COLLECTION_NAME)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )

    all_docs = []
    all_ids = []
    all_meta = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        rel = file_path.relative_to(KB_DIR).as_posix()
        for idx, chunk in enumerate(chunks):
            all_docs.append(chunk)
            all_ids.append(f"{rel}-{idx}-{uuid.uuid4().hex[:8]}")
            all_meta.append({"source": rel, "chunk_index": idx})

    if not all_docs:
        raise RuntimeError("No text chunks created from KB files")

    batch_size = 100
    for i in range(0, len(all_docs), batch_size):
        collection.add(
            documents=all_docs[i:i + batch_size],
            ids=all_ids[i:i + batch_size],
            metadatas=all_meta[i:i + batch_size],
        )

    print(f"Ingested {len(all_docs)} chunks from {len(files)} files into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest AtlaOps KB files into ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate KB collection before ingest")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (fallback: OPENAI_API_KEY env)")
    args = parser.parse_args()

    api_key = args.api_key or __import__("os").environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for embeddings")

    ingest(reset=args.reset, api_key=api_key)
