from rag.retriever import Retriever
from llm.client import LLMClient
from llm.cost_tracker import CostTracker


CLINICAL_SYSTEM_PROMPT = """You are a clinical decision support assistant for 
frontline TB health workers in low-resource settings.

You answer questions using ONLY the guideline context provided below. 
You do not use your general training knowledge for clinical recommendations.

Structure every response exactly as follows:

RECOMMENDED ACTION:
[Clear, specific recommendation in 1-3 sentences]

SUPPORTING REGIMEN OR PROTOCOL:
[Drug names, doses, duration where relevant]

RED FLAGS / WHEN TO REFER:
[Conditions that require escalation or specialist input]

UNCERTAINTY:
[What this guidance does NOT cover, or where clinical judgment is needed]

SOURCES:
[List the source documents and pages you drew from]

---
Always end with: "Confirm with current national guidelines and supervising 
clinician before acting on this guidance."
"""


class RAGPipeline:
    """
    Connects retrieval to generation.

    Design decision: the system prompt enforces a fixed output
    structure rather than letting the LLM choose its format.

    Why this matters for clinical tools:
    - Predictable structure makes responses scannable under time pressure
    - RED FLAGS section ensures safety-critical information is never
      buried in prose
    - UNCERTAINTY section is non-negotiable — a tool that doesn't
      surface its own limits is dangerous in clinical settings
    - SOURCES section makes the grounding auditable

    This structure also makes automated evaluation tractable:
    each section can be scored independently.
    """

    def __init__(self,
                 provider: str = None,
                 model: str = None,
                 top_k: int = 4,
                 cost_tracker: CostTracker = None):
        self.retriever = Retriever(top_k=top_k)
        self.llm = LLMClient(
            provider=provider,
            model=model,
            cost_tracker=cost_tracker or CostTracker()
        )
        self.system_prompt = CLINICAL_SYSTEM_PROMPT
    def _build_user_message(self, question: str, context: str) -> str:
        """Builds the user message with context prepended."""
        return f"""GUIDELINE CONTEXT:
    {context}

    ---

    CLINICAL QUESTION:
    {question}

    Answer using only the guideline context above.
    If the context does not contain enough information to answer safely,
    say so explicitly in the UNCERTAINTY section."""

    def query(self, user_question: str) -> dict:
        """
        Full RAG pipeline for a single clinical query.

        Returns a dict with:
        - response: structured clinical guidance
        - chunks: the retrieved guideline passages
        - context: formatted context sent to LLM
        - cost: token usage and cost for this query
        """

        # Step 1: Retrieve relevant chunks
        chunks = self.retriever.retrieve(user_question)
        context = self.retriever.format_context(chunks)

        # Step 2: Build the user message with context prepended
        user_message = f"""GUIDELINE CONTEXT:
{context}

---

CLINICAL QUESTION:
{user_question}

Answer using only the guideline context above. 
If the context does not contain enough information to answer safely, 
say so explicitly in the UNCERTAINTY section."""

        # Step 3: Call the LLM
        response = self.llm.complete(
            system_prompt=CLINICAL_SYSTEM_PROMPT,
            user_message=user_message
        )

        return {
            "response": response["text"],
            "chunks": chunks,
            "context": context,
            "input_tokens": response["input_tokens"],
            "output_tokens": response["output_tokens"],
            "cost_usd": response["cost_usd"],
            "provider": response["provider"],
            "model": response["model"]
        }


if __name__ == "__main__":
    print("=== RAG Pipeline Test ===\n")

    pipeline = RAGPipeline()

    # Test 1: Standard first-line case
    print("TEST 1: Standard new patient query")
    print("-" * 40)
    question = "What treatment should a new TB patient receive?"
    result = pipeline.query(question)

    print(f"Question: {question}\n")
    print(f"Response:\n{result['response']}\n")
    print(f"Tokens: {result['input_tokens']} in, {result['output_tokens']} out")
    print(f"Cost: ${result['cost_usd']:.4f}")
    print(f"Sources used: {[c['source'] for c in result['chunks']]}\n")

    # Test 2: Edge case — drug resistance
    print("\nTEST 2: Drug resistance edge case")
    print("-" * 40)
    question2 = ("Patient previously treated for TB, "
                 "now presenting with recurrence. "
                 "What are the key considerations?")
    result2 = pipeline.query(question2)

    print(f"Question: {question2}\n")
    print(f"Response:\n{result2['response']}\n")
    print(f"Tokens: {result2['input_tokens']} in, {result2['output_tokens']} out")
    print(f"Cost: ${result2['cost_usd']:.4f}")

    # Session summary
    print("\n" + "=" * 40)
    print("SESSION SUMMARY")
    print("=" * 40)
    summary = pipeline.llm.cost_tracker.summary()
    print(f"Total queries:  {summary['queries']}")
    print(f"Total tokens:   {summary['total_tokens']}")
    print(f"Total cost:     {summary['cost_display']}")
    print(f"Ollama equivalent: $0.0000")