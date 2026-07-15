from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions


PROJECT_ROOT = Path(__file__).parent.parent
CHROMA_DIR = PROJECT_ROOT / "knowledge_base" / "chroma_db"
COLLECTION_NAME = "who_tb_guidelines"


class Retriever:
    """
    Retrieves relevant chunks from ChromaDB for a given clinical query.

    Design decision: retrieval is separated from the LLM call
    deliberately. This means you can test and evaluate retrieval
    quality independently of response quality. A common failure
    mode in RAG systems is poor retrieval — the LLM gives a bad
    answer not because it reasoned poorly but because it never
    received the right context. Keeping these separate makes that
    diagnosable.

    Embedding model: mxbai-embed-large via Ollama (local).
    The query is embedded using the same model as the documents.
    Using different models for ingestion and retrieval is a common
    mistake that silently degrades performance.
    """

    def __init__(self, top_k: int = 4):
        """
        top_k: number of chunks to retrieve per query.

        Why 4: enough context for a complete clinical recommendation
        without exceeding practical token limits. Tunable in
        config/app_config.yaml.
        """
        self.top_k = top_k
        self.collection = self._load_collection()

    def _load_collection(self) -> chromadb.Collection:
        embedding_function = embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="mxbai-embed-large",
            timeout=120
        )

        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        return client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )

    def retrieve(self, query: str) -> list[dict]:
        """
        Takes a clinical query string.
        Returns a list of the top_k most relevant chunks.

        Each chunk contains:
        - text: the guideline passage
        - source: the PDF filename it came from
        - page: the page number
        - distance: similarity score (lower = more similar)
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=self.top_k
        )

        chunks = []
        for i in range(len(results["documents"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("filename", "unknown"),
                "page": results["metadatas"][0][i].get("page", 0),
                "distance": results["distances"][0][i]
            })

        return chunks

    def format_context(self, chunks: list[dict]) -> str:
        """
        Formats retrieved chunks into a single context string
        for the LLM prompt.

        Each chunk is labeled with its source so the LLM can
        cite it in the response.
        """
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk['source']}, Page {chunk['page']}]\n"
                f"{chunk['text']}"
            )

        return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    print("=== Retriever Test ===\n")

    retriever = Retriever(top_k=4)

    # Test with a realistic clinical query
    query = "What is the recommended treatment for new TB patients?"
    print(f"Query: {query}\n")

    chunks = retriever.retrieve(query)

    print(f"Retrieved {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"  Source: {chunk['source']}, Page {chunk['page']}")
        print(f"  Distance: {chunk['distance']:.4f}")
        print(f"  Text preview: {chunk['text'][:150]}...")
        print()

    print("--- Formatted context (first 500 chars) ---")
    context = retriever.format_context(chunks)
    print(context[:500])