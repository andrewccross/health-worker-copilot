import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Paths — all relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
GUIDELINES_DIR = PROJECT_ROOT / "knowledge_base" / "who_tb_guidelines"
CHROMA_DIR = PROJECT_ROOT / "knowledge_base" / "chroma_db"
COLLECTION_NAME = "who_tb_guidelines"


def load_pdfs(guidelines_dir: Path) -> list:
    """
    Load all PDFs from the guidelines directory.
    
    Design decision: we use PyMuPDF (fitz) rather than pypdf because
    WHO documents contain complex layouts, tables, and formatting.
    PyMuPDF handles these more reliably and preserves more text structure.
    
    Returns a list of LangChain Document objects, each with:
    - page_content: the extracted text
    - metadata: source filename and page number
    """
    documents = []
    pdf_files = list(guidelines_dir.glob("*.pdf"))
    
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {guidelines_dir}. "
            f"Please download WHO TB guidelines first."
        )
    
    print(f"Found {len(pdf_files)} PDF files:")
    
    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()
        documents.extend(docs)
        print(f"    → {len(docs)} pages extracted")
    
    print(f"\nTotal pages loaded: {len(documents)}")
    return documents


def chunk_documents(documents: list) -> list:
    """
    Split documents into chunks suitable for embedding and retrieval.
    
    Design decision: chunk_size=500, chunk_overlap=50
    
    Why 500 tokens: large enough to contain a complete clinical 
    recommendation with context, small enough that retrieval returns
    specific relevant passages rather than entire sections.
    
    Why overlap: 50-token overlap means a recommendation that spans
    a chunk boundary isn't lost. The overlap duplicates some text
    but ensures continuity.
    
    These values are set in config/app_config.yaml and can be tuned
    based on evaluation results.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_vector_store(chunks: list) -> chromadb.Collection:
    """
    Embed chunks and store in ChromaDB.
    
    Design decision: we use Ollama's mxbai-embed-large for embeddings.
    
    Why local embeddings: sending guideline text to an external 
    embedding API is unnecessary cost and a data sovereignty concern.
    mxbai-embed-large is a high-quality embedding model that runs
    entirely on your machine. This means ingestion has zero API cost
    and zero data transmission.
    
    The ChromaDB collection persists to disk at knowledge_base/chroma_db/.
    Once built, retrieval requires no re-embedding.
    """
    print(f"\nInitializing ChromaDB at: {CHROMA_DIR}")
    
    # Use Ollama for local embeddings — no API key needed
    embedding_function = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="mxbai-embed-large",
        timeout=120
    )
    
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # Delete existing collection if rebuilding
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection (rebuilding)")
    except Exception:
        pass
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Add chunks in batches to avoid memory issues
    batch_size = 10
    total_chunks = len(chunks)
    
    print(f"\nEmbedding {total_chunks} chunks using mxbai-embed-large (local)...")
    print("This runs entirely on your machine — no data leaves your system.")
    print("First run takes several minutes. Subsequent queries are instant.\n")
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        texts = [doc.page_content for doc in batch]
        metadatas = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0),
                "filename": Path(
                    doc.metadata.get("source", "unknown")
                ).name
            }
            for doc in batch
        ]
        
        collection.add(ids=ids, documents=texts, metadatas=metadatas)
        
        progress = min(i + batch_size, total_chunks)
        print(f"  Embedded {progress}/{total_chunks} chunks")
    
    print(f"\nDone. {total_chunks} chunks stored in ChromaDB.")
    print(f"Location: {CHROMA_DIR}")
    return collection


def main():
    print("=== Health Worker AI Copilot — Knowledge Base Ingestion ===\n")
    
    # Verify Ollama is running
    print("Step 1: Loading PDFs")
    documents = load_pdfs(GUIDELINES_DIR)
    
    print("\nStep 2: Chunking documents")
    chunks = chunk_documents(documents)
    
    print("\nStep 3: Embedding and storing in ChromaDB")
    collection = build_vector_store(chunks)
    
    print("\n=== Ingestion complete ===")
    print(f"Collection '{COLLECTION_NAME}' ready for querying.")
    print("\nNext step: run python -m rag.retriever to test retrieval")


if __name__ == "__main__":
    main()