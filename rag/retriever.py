from pathlib import Path
import chromadb
import tempfile
import os
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).parent.parent
CHROMA_DIR = PROJECT_ROOT / "knowledge_base" / "chroma_db"
COLLECTION_NAME = "who_tb_guidelines"


def get_embedding_function():
    """
    Embedding model: sentence-transformers all-MiniLM-L6-v2.

    Design decision: we use sentence-transformers rather than
    Ollama mxbai-embed-large for embeddings.

    Alternatives considered:

    Option A (chosen): sentence-transformers all-MiniLM-L6-v2
    - Runs locally, no external service dependency
    - Works identically on local and cloud deployment
    - Slightly lower semantic accuracy than larger models
    - Zero cost at ingestion and query time
    - 384-dimensional vectors, fast on CPU

    Option B: Ollama mxbai-embed-large
    - Higher quality embeddings for clinical text
    - Requires Ollama running as a background service
    - Not available on cloud deployment (Streamlit Community Cloud
      has no Ollama installation)
    - Would require separate embedding strategy per environment,
      and ChromaDB built with one model cannot be queried with
      another without full re-ingestion

    Option C: OpenAI text-embedding-3-small via API
    - High quality, managed service
    - Requires OpenAI API key and incurs per-token cost
    - ~$0.004 for full ingestion of this knowledge base
    - Adds external dependency for a sovereignty-focused tool
    - Not chosen: contradicts the local-first design principle

    Consistency rule: ingestion and retrieval MUST use the same
    embedding model. Mixing models silently degrades retrieval
    quality because vectors are not comparable across models.
    This is a common and hard-to-debug failure mode in RAG systems.
    """
    from chromadb.utils.embedding_functions import (
        SentenceTransformerEmbeddingFunction
    )
    return SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


class Retriever:
    """
    Retrieves relevant chunks from ChromaDB.

    Supports two collections simultaneously:
    - Base WHO guidelines (always present, persisted to disk)
    - User-uploaded document (optional, in-memory, session-scoped)

    Design decision: uploaded documents augment rather than replace
    the base knowledge. Both are searched and results merged by
    relevance score. When sources conflict, both are surfaced and
    the LLM's system prompt instructs it to flag the discrepancy.
    """

    def __init__(self, top_k: int = 4):
        self.top_k = top_k
        self.embedding_function = get_embedding_function()
        self.base_collection = self._load_base_collection()
        self.upload_collection = None
        self._chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )
        self._memory_client = chromadb.EphemeralClient()

    def _load_base_collection(self) -> chromadb.Collection:
        """
        Load existing collection, rebuilding if not found.

        Design decision: if the collection doesn't exist (e.g. on a
        fresh cloud deployment where the SQLite file wasn't committed),
        we rebuild it from the source PDFs rather than crashing.

        This makes the app self-healing at the cost of a longer first
        startup on cloud deployments without a pre-built database.
        """
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        try:
            return client.get_collection(name=COLLECTION_NAME)
        except Exception:
            print(
                f"Collection '{COLLECTION_NAME}' not found. "
                f"Rebuilding from source PDFs..."
            )
            return self._rebuild_collection(client)

    def _rebuild_collection(self, client) -> chromadb.Collection:
        """
        Rebuilds ChromaDB from source PDFs.
        Called automatically if collection is missing.
        """
        from langchain_community.document_loaders import PyMuPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        guidelines_dir = PROJECT_ROOT / "knowledge_base" / "who_tb_guidelines"
        pdf_files = list(guidelines_dir.glob("*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(
                f"No PDFs found in {guidelines_dir}. "
                f"Cannot rebuild knowledge base."
            )

        # Load PDFs
        documents = []
        for pdf_path in pdf_files:
            loader = PyMuPDFLoader(str(pdf_path))
            documents.extend(loader.load())

        # Chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(documents)

        # Create collection
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

        # Add in batches
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            collection.add(
                ids=[f"chunk_{i + j}" for j in range(len(batch))],
                documents=[doc.page_content for doc in batch],
                metadatas=[
                    {
                        "source": doc.metadata.get("source", "unknown"),
                        "page": doc.metadata.get("page", 0),
                        "filename": Path(
                            doc.metadata.get("source", "unknown")
                        ).name
                    }
                    for doc in batch
                ]
            )

        print(f"Rebuilt collection with {len(chunks)} chunks.")
        return collection

    def add_uploaded_document(self, pdf_bytes: bytes,
                               filename: str) -> int:
        """
        Ingests an uploaded PDF into a temporary in-memory collection.
        Returns the number of chunks added.

        This collection is never written to disk.
        It exists only for the current session.
        """

        # Write bytes to a temp file so PyMuPDF can read it
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False
        ) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            # Load and chunk the PDF
            loader = PyMuPDFLoader(tmp_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            chunks = splitter.split_documents(documents)

            # Create or replace in-memory collection
            try:
                self._memory_client.delete_collection("uploaded_doc")
            except Exception:
                pass

            self.upload_collection = self._memory_client.create_collection(
                name="uploaded_doc",
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )

            # Add chunks
            ids = [f"upload_chunk_{i}" for i in range(len(chunks))]
            texts = [doc.page_content for doc in chunks]
            metadatas = [
                {
                    "source": filename,
                    "page": doc.metadata.get("page", 0),
                    "filename": filename,
                    "collection": "uploaded"
                }
                for doc in chunks
            ]

            self.upload_collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )

            return len(chunks)

        finally:
            os.unlink(tmp_path)

    def clear_uploaded_document(self):
        """Remove the uploaded document collection."""
        self.upload_collection = None
        try:
            self._memory_client.delete_collection("uploaded_doc")
        except Exception:
            pass

    def retrieve(self, query: str) -> list[dict]:
        """
        Retrieves from base collection, and uploaded collection
        if present. Merges and deduplicates results by relevance.
        """
        # Always retrieve from base WHO guidelines
        base_results = self._query_collection(
            self.base_collection, query, self.top_k
        )

        if self.upload_collection is None:
            return base_results

        # Also retrieve from uploaded document
        upload_results = self._query_collection(
            self.upload_collection, query, self.top_k
        )

        # Merge and sort by distance (lower = more relevant)
        combined = base_results + upload_results
        combined.sort(key=lambda x: x["distance"])

        # Return top_k from combined results
        return combined[:self.top_k]

    def _query_collection(self, collection,
                       query: str,
                       n: int) -> list[dict]:
        # Embed the query manually using our embedding function
        # This bypasses ChromaDB's internal embedding to avoid
        # embedding function conflict errors
        query_embedding = self.embedding_function([query])
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n
        )

        chunks = []
        for i in range(len(results["documents"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i].get(
                    "filename", "unknown"
                ),
                "page": results["metadatas"][0][i].get("page", 0),
                "distance": results["distances"][0][i],
                "collection": results["metadatas"][0][i].get(
                    "collection", "base"
                )
            })

        return chunks

    def format_context(self, chunks: list[dict]) -> str:
        """
        Formats retrieved chunks into context for the LLM.
        Uploaded document chunks are labeled distinctly so the
        LLM knows to flag conflicts with the WHO base guidelines.
        """
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            collection_label = (
                "UPLOADED NATIONAL GUIDELINE"
                if chunk.get("collection") == "uploaded"
                else "WHO GLOBAL GUIDELINE"
            )
            context_parts.append(
                f"[Source {i} — {collection_label}: "
                f"{chunk['source']}, Page {chunk['page']}]\n"
                f"{chunk['text']}"
            )

        return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    print("=== Retriever Test ===\n")
    retriever = Retriever(top_k=4)
    query = "What is the recommended treatment for new TB-affected people?"
    chunks = retriever.retrieve(query)
    print(f"Retrieved {len(chunks)} chunks from base collection\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk['source']}, "
              f"Page {chunk['page']}, "
              f"Distance: {chunk['distance']:.4f}")