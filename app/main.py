import streamlit as st
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.pipeline import RAGPipeline
from llm.cost_tracker import CostTracker


# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Health Worker AI Copilot",
    page_icon="🏥",
    layout="wide"
)


# ── Session state initialisation ───────────────────────────────
# Streamlit reruns the entire script on every interaction.
# Session state persists values across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

if "cost_tracker" not in st.session_state:
    st.session_state.cost_tracker = CostTracker()

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "query_count" not in st.session_state:
    st.session_state.query_count = 0


# ── Constants ──────────────────────────────────────────────────
MAX_QUERIES = 10  # Demo safety limit

WELCOME_MESSAGE = """
**Welcome to the Health Worker AI Copilot.**

This tool provides clinical decision support for TB and infectious 
disease cases, grounded in WHO treatment guidelines.

**How to use:**
- Describe a patient case in plain language
- Include relevant details: symptoms, test results, treatment history
- Receive structured guidance with sources cited

**Example query:**
*"35-year-old patient, positive sputum smear, no prior TB treatment. 
What regimen should I start?"*

---
⚠️ This tool supports clinical decision-making. It does not replace 
a supervising clinician or current national guidelines.
"""


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    st.subheader("Model")
    provider = st.selectbox(
        "LLM Provider",
        options=["claude"],
        index=0,
        help="Additional providers (OpenAI, Ollama) coming in Stage 5"
    )

    st.divider()

    st.subheader("💰 Session Cost")
    summary = st.session_state.cost_tracker.summary()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", summary["queries"])
        st.metric("Total tokens", summary["total_tokens"])
    with col2:
        st.metric("API cost", summary["cost_display"])
        st.metric("Ollama cost", "$0.0000")

    st.divider()

    st.subheader("🔒 Data Sovereignty")
    if provider == "claude":
        st.error("● Data leaves facility\n(Claude API)")
    else:
        st.success("● Data stays local\n(Ollama)")

    st.divider()

    queries_remaining = MAX_QUERIES - st.session_state.query_count
    st.caption(f"Demo queries remaining: {queries_remaining}/{MAX_QUERIES}")

    if st.button("Reset session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.cost_tracker.reset()
        st.session_state.query_count = 0
        st.session_state.pipeline = None
        st.rerun()


# ── Main content ───────────────────────────────────────────────
st.title("🏥 Health Worker AI Copilot")
st.caption("Clinical decision support grounded in WHO TB treatment guidelines")

st.divider()

# Display welcome message on first load
if not st.session_state.messages:
    st.markdown(WELCOME_MESSAGE)

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "meta" in message:
            meta = message["meta"]
            st.caption(
                f"Sources: {', '.join(set(meta['sources']))} | "
                f"Tokens: {meta['input_tokens']} in, "
                f"{meta['output_tokens']} out | "
                f"Cost: ${meta['cost_usd']:.4f}"
            )

# ── Query input ────────────────────────────────────────────────
if st.session_state.query_count >= MAX_QUERIES:
    st.warning(
        "Demo query limit reached. "
        "Click 'Reset session' in the sidebar to continue."
    )
else:
    user_input = st.chat_input(
        "Describe a patient case or ask a clinical question..."
    )

    if user_input:
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        # Initialise pipeline if needed
        if st.session_state.pipeline is None:
            with st.spinner("Initialising knowledge base..."):
                st.session_state.pipeline = RAGPipeline(
                    provider=provider,
                    cost_tracker=st.session_state.cost_tracker
                )

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving guidelines and generating response..."):
                try:
                    result = st.session_state.pipeline.query(user_input)

                    st.markdown(result["response"])
                    st.caption(
                        f"Sources: "
                        f"{', '.join(set(c['source'] for c in result['chunks']))} | "
                        f"Tokens: {result['input_tokens']} in, "
                        f"{result['output_tokens']} out | "
                        f"Cost: ${result['cost_usd']:.4f}"
                    )

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["response"],
                        "meta": {
                            "sources": [c["source"] for c in result["chunks"]],
                            "input_tokens": result["input_tokens"],
                            "output_tokens": result["output_tokens"],
                            "cost_usd": result["cost_usd"]
                        }
                    })

                    st.session_state.query_count += 1

                    # Refresh sidebar cost display
                    st.rerun()

                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.info(
                        "Check that Ollama is running and your "
                        "API key is set in .env"
                    )