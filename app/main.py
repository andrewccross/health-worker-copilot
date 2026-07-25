import streamlit as st
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.pipeline import RAGPipeline
from llm.cost_tracker import CostTracker
from app.comparison import (
    run_comparison,
    render_comparison_results,
    render_comparison_loading
)

import os

# Detect deployment environment
# IS_CLOUD_DEPLOYMENT is set in Streamlit Cloud secrets
# Never set locally, so defaults to False
IS_CLOUD = os.getenv("IS_CLOUD_DEPLOYMENT", "false").lower() == "true"


# ── Cached pipeline ────────────────────────────────────────────
@st.cache_resource
def get_pipeline(provider: str, model: str) -> RAGPipeline:
    """
    Initialize the RAG pipeline once and cache it.
    Keyed by provider and model — switching providers creates a
    new cached instance without reinitializing unnecessarily.
    """
    return RAGPipeline(provider=provider, model=model)


# ── Page config — must be first Streamlit call ─────────────────
st.set_page_config(
    page_title="Health Worker AI Copilot",
    page_icon="🏥",
    layout="wide"
)


# ── Session state ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "comparison_mode" not in st.session_state:
    st.session_state.comparison_mode = False

if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None

if "cost_tracker" not in st.session_state:
    st.session_state.cost_tracker = CostTracker()

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None


# ── Constants ──────────────────────────────────────────────────
MAX_QUERIES = 10

WELCOME_MESSAGE = """
**Welcome to the Health Worker AI Copilot.**

This tool provides clinical decision support for TB and infectious 
disease cases, grounded in WHO treatment guidelines.

**How to use:**
- Describe a person's health case in plain language
- Include relevant details: symptoms, test results, treatment history
- Receive structured guidance with sources cited

**Example query:**
*"35-year-old person seeking care, positive sputum smear, no prior 
TB treatment. What regimen should I start?"*

---
⚠️ This tool supports clinical decision-making. It does not replace 
a supervising clinician or current national guidelines.
"""


# ── Pipeline warmup ────────────────────────────────────────────
# Show a loading screen on first visit, then rerun into the full app.
# This must come after set_page_config and session state,
# but before the sidebar — otherwise sidebar variables are undefined
# when the rerun fires.

if not st.session_state.pipeline_ready:
    st.markdown(
        """
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 60vh;
            text-align: center;
        ">
            <h2 style="color: #ffffff;">
                🏥 Health Worker AI Copilot
            </h2>
            <p style="color: #cccccc; font-size: 1.1em;">
                Loading WHO TB guidelines knowledge base...
            </p>
            <p style="color: #aaaaaa; font-size: 0.9em;">
                Initialising local embedding model (mxbai-embed-large)
            </p>
            <p style="color: #888888; font-size: 0.8em; margin-top: 1em;">
                First load takes ~20 seconds. Subsequent queries are fast.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.spinner("Loading knowledge base..."):
        get_pipeline("claude", "claude-sonnet-4-6")

    st.session_state.pipeline_ready = True
    st.rerun()


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    st.success("✓ Knowledge base ready", icon="🗂️")

    st.subheader("Model")

    # On cloud deployment, hide Ollama — it requires local installation
    available_providers = (
        ["claude", "openai"] if IS_CLOUD
        else ["claude", "openai", "ollama"]
    )

    provider = st.selectbox(
        "LLM Provider",
        options=available_providers,
        index=0
    )

    model_options = {
        "claude": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        "openai": ["gpt-4o", "gpt-4o-mini"],
        "ollama": ["llama3.2", "llama2", "mistral"]
    }

    model = st.selectbox(
        "Model",
        options=model_options[provider],
        index=0
    )

    if provider == "claude":
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-... (or set in .env)",
            help="Leave blank to use key from .env file"
        )
        if api_key:
            import os
            os.environ["ANTHROPIC_API_KEY"] = api_key

    elif provider == "openai":
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-... (or set in .env)",
            help="Leave blank to use key from .env file"
        )
        if api_key:
            import os
            os.environ["OPENAI_API_KEY"] = api_key

    elif provider == "ollama":
        if IS_CLOUD:
            st.warning(
                "Ollama requires local deployment.\n"
                "Clone the repo to use local models.\n"
                "See README for setup instructions."
            )
        else:
            st.info(
                "Ollama runs locally.\n"
                "No API key required.\n"
                "Make sure `ollama serve` is running."
            )

    st.divider()

    st.subheader("🔀 Comparison Mode")
    st.session_state.comparison_mode = st.toggle(
        "Compare two providers",
        value=st.session_state.comparison_mode,
        help="Submit one query and see responses from two providers side by side"
    )

    if st.session_state.comparison_mode:
        st.caption("Provider A is selected above.")
        
        provider_b = st.selectbox(
            "Provider B",
            options=(
                ["claude", "openai"] if IS_CLOUD
                else ["claude", "openai", "ollama"]
            ),
            index=1,
            key="provider_b"
        )

        model_b_options = {
            "claude": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
            "openai": ["gpt-4o", "gpt-4o-mini"],
            "ollama": ["llama3.2", "llama2", "mistral"]
        }

        model_b = st.selectbox(
            "Model B",
            options=model_b_options[provider_b],
            index=0,
            key="model_b"
        )

        if provider_b == "openai":
            api_key_b = st.text_input(
                "OpenAI API Key (Provider B)",
                type="password",
                placeholder="sk-...",
                key="api_key_b"
            )
            if api_key_b:
                import os
                os.environ["OPENAI_API_KEY"] = api_key_b

        elif provider_b == "ollama" and not IS_CLOUD:
            st.info("Ollama runs locally. Make sure ollama serve is running.")

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

    sovereignty = {
        "claude": {
            "status": "error",
            "rung": "Rung 1: Cloud API",
            "message": "● Data leaves facility",
            "detail": "Queries sent to Anthropic API"
        },
        "openai": {
            "status": "error",
            "rung": "Rung 1: Cloud API",
            "message": "● Data leaves facility",
            "detail": "Queries sent to OpenAI API"
        },
        "ollama": {
            "status": "success",
            "rung": "Rung 4: Fully Local",
            "message": "● Data stays on device",
            "detail": "No data transmission"
        }
    }

    s = sovereignty[provider]
    if s["status"] == "error":
        st.error(f"{s['message']}\n{s['detail']}")
    else:
        st.success(f"{s['message']}\n{s['detail']}")

    st.caption(s["rung"])

    st.divider()

    queries_remaining = MAX_QUERIES - st.session_state.query_count
    st.caption(
        f"Demo queries remaining: {queries_remaining}/{MAX_QUERIES}"
    )

    if st.button("Reset session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.cost_tracker.reset()
        st.session_state.query_count = 0
        st.session_state.uploaded_filename = None
        st.session_state.comparison_results = None
        st.rerun()

    st.divider()

    st.subheader("📄 National Guidelines")
    st.caption(
        "Upload a country-specific guideline to augment "
        "the WHO base knowledge for this session."
    )

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Document is processed locally and not stored permanently."
    )

    if uploaded_file is not None:
        if st.session_state.uploaded_filename != uploaded_file.name:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    pipeline = get_pipeline(provider, model)
                    if provider == "ollama" and IS_CLOUD:
                        st.warning(
                            "Ollama is not available in the cloud demo. "
                            "Select Claude or clone the repo to run locally."
                        )
                        st.stop()
                    pdf_bytes = uploaded_file.read()
                    chunk_count = (
                        pipeline.retriever.add_uploaded_document(
                            pdf_bytes, uploaded_file.name
                        )
                    )
                    st.session_state.uploaded_filename = (
                        uploaded_file.name
                    )
                    st.success(
                        f"✓ {uploaded_file.name}\n"
                        f"{chunk_count} chunks added"
                    )
                except Exception as e:
                    st.error(f"Upload failed: {str(e)}")
        else:
            st.success(f"✓ {uploaded_file.name} active")

    elif st.session_state.uploaded_filename:
        pipeline = get_pipeline(provider, model)
        pipeline.retriever.clear_uploaded_document()
        st.session_state.uploaded_filename = None


# ── Main content ───────────────────────────────────────────────
st.title("🏥 Health Worker AI Copilot")
st.caption(
    "Clinical decision support grounded in WHO TB treatment guidelines"
)

st.divider()

if not st.session_state.messages:
    st.markdown(WELCOME_MESSAGE)

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

# ── Render stored comparison results ──────────────────────────
if (st.session_state.comparison_mode and
        st.session_state.comparison_results is not None):
    render_comparison_results(
        st.session_state.comparison_results
    )

# ── Query input ────────────────────────────────────────────────
if st.session_state.query_count >= MAX_QUERIES:
    st.warning(
        "Demo query limit reached. "
        "Click 'Reset session' in the sidebar to continue."
    )
else:
    user_input = st.chat_input(
        "Describe a person's health case or ask a clinical question..."
    )

    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        # ── Comparison mode ─────────────────────────────────────────
        if st.session_state.comparison_mode:
            provider_b = st.session_state.get("provider_b", "openai")
            model_b = st.session_state.get("model_b", "gpt-4o")

            with st.chat_message("assistant"):
                # Show skeleton immediately
                skeleton = st.empty()
                with skeleton.container():
                    render_comparison_loading(
                        provider_a=provider,
                        model_a=model,
                        provider_b=provider_b,
                        model_b=model_b,
                        sources=[]
                    )

                # Top-level spinner shows Streamlit is actively working
                with st.spinner(
                    f"Querying {provider} and {provider_b} — "
                    f"this takes 15-60 seconds..."
                ):
                    try:
                        results = run_comparison(
                            query=user_input,
                            provider_a=provider,
                            model_a=model,
                            provider_b=provider_b,
                            model_b=model_b,
                            get_pipeline_fn=get_pipeline
                        )

                        # Clear skeleton
                        skeleton.empty()

                        st.session_state.comparison_results = results

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": (
                                f"**Comparison: "
                                f"{provider.title()} vs "
                                f"{provider_b.title()}**"
                            ),
                            "meta": {
                                "sources": results["sources"],
                                "input_tokens": (
                                    results["a"]["input_tokens"] +
                                    results["b"]["input_tokens"]
                                ),
                                "output_tokens": (
                                    results["a"]["output_tokens"] +
                                    results["b"]["output_tokens"]
                                ),
                                "cost_usd": (
                                    results["a"]["cost_usd"] +
                                    results["b"]["cost_usd"]
                                )
                            }
                        })

                        st.session_state.query_count += 1
                        st.rerun()

                    except Exception as e:
                        skeleton.empty()
                        st.error(f"Comparison error: {str(e)}")

        else:
            # ── Standard mode ──────────────────────────────
            pipeline = get_pipeline(provider, model)
            pipeline.llm.cost_tracker = st.session_state.cost_tracker

            with st.chat_message("assistant"):
                try:
                    if provider == "claude":
                        with st.spinner("Retrieving guidelines..."):
                            chunks = pipeline.retriever.retrieve(
                                user_input
                            )
                            context = pipeline.retriever.format_context(
                                chunks
                            )
                            user_message = pipeline._build_user_message(
                                user_input, context
                            )

                        response_container = st.empty()
                        full_response = ""

                        for text_chunk in (
                            pipeline.llm._complete_claude_stream(
                                system_prompt=pipeline.system_prompt,
                                user_message=user_message,
                                max_tokens=1024
                            )
                        ):
                            full_response += text_chunk
                            response_container.markdown(
                                full_response + "▌"
                            )

                        response_container.markdown(full_response)
                        usage = pipeline.llm._last_usage

                        st.caption(
                            f"Sources: "
                            f"{', '.join(set(c['source'] for c in chunks))}"
                            f" | Tokens: {usage['input_tokens']} in, "
                            f"{usage['output_tokens']} out | "
                            f"Cost: ${usage['cost_usd']:.4f}"
                        )

                        result = {
                            "response": full_response,
                            "chunks": chunks,
                            "input_tokens": usage["input_tokens"],
                            "output_tokens": usage["output_tokens"],
                            "cost_usd": usage["cost_usd"]
                        }

                    else:
                        with st.spinner(
                            "Retrieving guidelines and generating "
                            "response..."
                        ):
                            result = pipeline.query(user_input)

                        st.markdown(result["response"])
                        st.caption(
                            f"Sources: "
                            f"{', '.join(set(c['source'] for c in result['chunks']))}"
                            f" | Tokens: {result['input_tokens']} in, "
                            f"{result['output_tokens']} out | "
                            f"Cost: ${result['cost_usd']:.4f}"
                        )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["response"],
                        "meta": {
                            "sources": [
                                c["source"] for c in result["chunks"]
                            ],
                            "input_tokens": result["input_tokens"],
                            "output_tokens": result["output_tokens"],
                            "cost_usd": result["cost_usd"]
                        }
                    })

                    st.session_state.query_count += 1
                    st.rerun()

                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.info(
                        "Check that Ollama is running and your "
                        "API key is set in .env"
                    )