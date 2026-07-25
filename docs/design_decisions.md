# Design Decisions

## Embedding Model Selection

**Decision:** sentence-transformers all-MiniLM-L6-v2

**Date:** July 2026

**Context:** The RAG pipeline requires an embedding model to convert
text chunks and queries into vectors for semantic search. Three
options were evaluated.

### Options considered

**Option A — sentence-transformers all-MiniLM-L6-v2 (chosen)**
- Runs entirely locally, no external service required
- Works identically on local machines and cloud deployment
- 384-dimensional vectors, fast inference on CPU
- Zero cost at ingestion and query time
- Slight quality reduction vs larger models on technical text

**Option B — Ollama mxbai-embed-large**
- Higher semantic accuracy on clinical and technical text
- Requires Ollama running as a background service
- Not available on Streamlit Community Cloud
- Would require separate embedding strategies per environment
- ChromaDB built with one model cannot be queried with another
  without full re-ingestion — a hard constraint

**Option C — OpenAI text-embedding-3-small**
- High quality managed embeddings
- Requires OpenAI API key
- ~$0.004 for full corpus ingestion (negligible cost)
- Adds external API dependency to a sovereignty-focused tool
- Contradicts the local-first design principle

### Decision rationale

Option A was chosen for deployment consistency. A system that
behaves identically in local and cloud environments is easier
to evaluate, debug, and trust. The quality tradeoff is real
but modest for English-language clinical queries.

A production deployment serving multilingual contexts (French,
Vietnamese, Swahili) should re-evaluate this decision — larger
embedding models show meaningfully better performance on
low-resource languages.

### Consistency rule

Ingestion and retrieval must always use the same embedding model.
This is enforced by the single `get_embedding_function()` in
`rag/retriever.py` that both `ingest.py` and `retriever.py` import.
Never change one without changing the other.

---

## LLM Provider Abstraction

**Decision:** single `LLMClient` class with deferred provider imports

**Context:** The tool needs to support three LLM providers (Claude,
OpenAI, Ollama) with a single interface. Provider imports are
deferred to `_setup_client()` rather than imported at module level.

**Rationale:** deferred imports mean the app doesn't crash on startup
if a provider's package is missing or its API key is absent. It only
fails when that provider is actually selected. This is important for
Rung 4 deployments where API keys will never exist.

---

## Streamlit vs FastAPI + React

**Decision:** Streamlit for the demo interface

**Rationale:** Streamlit allows rapid iteration and deployment from
a single Python file. For a portfolio demonstration targeting
non-engineering reviewers, the time-to-demo advantage outweighs
the architectural limitations.

**Known limitations:** Streamlit reruns the entire script on every
interaction, limiting fine-grained UI control. A production
deployment would use FastAPI + React for better performance,
offline support, and mobile responsiveness.

---

## ChromaDB vs FAISS vs Pinecone

**Decision:** ChromaDB with persistent local storage

**Rationale:**
- Runs locally, no managed service dependency
- Persists to disk — pre-built vector store committed to repo
- Clean Python API, easy to inspect and debug
- Migration path to Pinecone is one config change

**Tradeoff:** ChromaDB is slower than FAISS at large scale.
For a knowledge base of 2,151 chunks this is not meaningful.
At 100,000+ chunks, FAISS or a managed vector store would
be appropriate.