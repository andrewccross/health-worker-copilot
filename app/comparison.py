import streamlit as st
import time
from rag.pipeline import RAGPipeline, CLINICAL_SYSTEM_PROMPT
from llm.cost_tracker import CostTracker


def run_comparison(
    query: str,
    provider_a: str,
    model_a: str,
    provider_b: str,
    model_b: str,
    get_pipeline_fn
) -> dict:
    """
    Runs both providers and returns structured results.
    Does NOT render anything — rendering is handled separately.
    This separation means results can be stored in session state
    and rendered consistently on every rerun.
    """
    pipeline_a = get_pipeline_fn(provider_a, model_a)

    # Retrieve once, shared across both providers
    chunks = pipeline_a.retriever.retrieve(query)
    context = pipeline_a.retriever.format_context(chunks)
    user_message = pipeline_a._build_user_message(query, context)
    sources = list(set(c["source"] for c in chunks))

    results = {
        "query": query,
        "sources": sources,
        "chunks": chunks,
        "a": _run_provider(
            provider_a, model_a,
            user_message, get_pipeline_fn
        ),
        "b": _run_provider(
            provider_b, model_b,
            user_message, get_pipeline_fn
        )
    }

    return results


def _run_provider(
    provider: str,
    model: str,
    user_message: str,
    get_pipeline_fn
) -> dict:
    """
    Runs a single provider silently and returns result dict.
    No Streamlit rendering happens here.
    """
    pipeline = get_pipeline_fn(provider, model)
    tracker = CostTracker()
    pipeline.llm.cost_tracker = tracker
    start_time = time.time()

    try:
        response = pipeline.llm.complete(
            system_prompt=CLINICAL_SYSTEM_PROMPT,
            user_message=user_message,
            max_tokens=1024
        )
        elapsed = time.time() - start_time

        return {
            "provider": provider,
            "model": model,
            "response": response["text"],
            "input_tokens": response["input_tokens"],
            "output_tokens": response["output_tokens"],
            "cost_usd": response["cost_usd"],
            "elapsed": elapsed,
            "error": None
        }

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "provider": provider,
            "model": model,
            "response": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "elapsed": elapsed,
            "error": str(e)
        }

def render_comparison_loading(
    provider_a: str,
    model_a: str,
    provider_b: str,
    model_b: str,
    sources: list
):
    """
    Renders the comparison skeleton with active spinners
    while generation is running.
    """
    st.caption("Retrieving context and querying both providers...")

    st.divider()

    # Metrics skeleton
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric(f"🔵 {provider_a.title()}", "...")
    with col2:
        st.metric("Tokens in", "...")
    with col3:
        st.metric("Time", "...")
    with col4:
        st.metric(f"🟢 {provider_b.title()}", "...")
    with col5:
        st.metric("Tokens in", "...")
    with col6:
        st.metric("Time", "...")

    # Sovereignty skeleton
    sov_a, sov_b = st.columns(2)
    with sov_a:
        st.info(f"● {provider_a.title()} — querying...")
    with sov_b:
        st.info(f"● {provider_b.title()} — querying...")

    st.divider()

    # Response columns with active spinners
    resp_a, resp_b = st.columns(2)
    with resp_a:
        st.markdown(f"**🔵 {provider_a.title()} / {model_a}**")
        st.spinner(f"Generating {provider_a} response...")
        with st.container(border=True):
            st.markdown(
                """
                <div style="
                    color: #888;
                    font-size: 0.9em;
                    padding: 1em;
                    text-align: center;
                ">
                ⏳ Generating response...
                </div>
                """,
                unsafe_allow_html=True
            )

    with resp_b:
        st.markdown(f"**🟢 {provider_b.title()} / {model_b}**")
        st.spinner(f"Generating {provider_b} response...")
        with st.container(border=True):
            st.markdown(
                """
                <div style="
                    color: #888;
                    font-size: 0.9em;
                    padding: 1em;
                    text-align: center;
                ">
                ⏳ Generating response...
                </div>
                """,
                unsafe_allow_html=True
            )

def render_comparison_results(results: dict):
    """
    Renders stored comparison results in a stable side-by-side layout.
    Called on every rerun from session state — never during generation.
    This ensures the layout is always consistent regardless of
    content length.
    """
    result_a = results["a"]
    result_b = results["b"]

    # Sources header
    st.caption(
        f"Both providers received identical context from: "
        f"{', '.join(results['sources'])}"
    )

    st.divider()

    # ── Metrics row (always side by side) ─────────────────────
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric(
            f"🔵 {result_a['provider'].title()}",
            f"${result_a['cost_usd']:.4f}"
        )
    with col2:
        st.metric(
            "Tokens in",
            f"{result_a['input_tokens']}"
        )
    with col3:
        st.metric(
            "Time",
            f"{result_a['elapsed']:.1f}s"
        )
    with col4:
        st.metric(
            f"🟢 {result_b['provider'].title()}",
            f"${result_b['cost_usd']:.4f}"
        )
    with col5:
        st.metric(
            "Tokens in",
            f"{result_b['input_tokens']}"
        )
    with col6:
        st.metric(
            "Time",
            f"{result_b['elapsed']:.1f}s"
        )

    # ── Sovereignty row (always side by side) ──────────────────
    sov_a, sov_b = st.columns(2)
    with sov_a:
        if result_a["provider"] == "ollama":
            st.success("● Data stayed on device")
        else:
            st.error("● Data left facility")
    with sov_b:
        if result_b["provider"] == "ollama":
            st.success("● Data stayed on device")
        else:
            st.error("● Data left facility")

    st.divider()

    # ── Response text (side by side, scrollable) ───────────────
    resp_a, resp_b = st.columns(2)

    with resp_a:
        st.markdown(
            f"**🔵 {result_a['provider'].title()} "
            f"/ {result_a['model']}**"
        )
        if result_a["error"]:
            st.error(f"Error: {result_a['error']}")
        else:
            st.markdown(result_a["response"])

    with resp_b:
        st.markdown(
            f"**🟢 {result_b['provider'].title()} "
            f"/ {result_b['model']}**"
        )
        if result_b["error"]:
            st.error(f"Error: {result_b['error']}")
        else:
            st.markdown(result_b["response"])

    st.divider()

    # ── Feedback row ───────────────────────────────────────────
    st.caption("Which response was more useful?")
    fb1, fb2, fb3 = st.columns(3)

    with fb1:
        if st.button(
            f"👈 {result_a['provider'].title()} was better",
            use_container_width=True,
            key="fb_a"
        ):
            _save_feedback(results, "a")
            st.success("Feedback recorded.")
    with fb2:
        if st.button(
            "👐 Equal",
            use_container_width=True,
            key="fb_equal"
        ):
            _save_feedback(results, "equal")
            st.success("Feedback recorded.")
    with fb3:
        if st.button(
            f"👉 {result_b['provider'].title()} was better",
            use_container_width=True,
            key="fb_b"
        ):
            _save_feedback(results, "b")
            st.success("Feedback recorded.")


def _save_feedback(results: dict, winner: str):
    if "comparison_feedback" not in st.session_state:
        st.session_state.comparison_feedback = []

    st.session_state.comparison_feedback.append({
        "query": results["query"],
        "provider_a": results["a"]["provider"],
        "model_a": results["a"]["model"],
        "provider_b": results["b"]["provider"],
        "model_b": results["b"]["model"],
        "winner": winner,
        "sources": results["sources"],
        "cost_a": results["a"]["cost_usd"],
        "cost_b": results["b"]["cost_usd"]
    })