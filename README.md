# Health Worker AI Copilot

**Demonstration RAG-powered clinical decision support for frontline TB and 
infectious disease health workers in low-resource settings.**

Built by [Andrew Cross](https://www.linkedin.com/in/andrewccross/).

[**→ Live Demo**](https://health-worker-copilot.streamlit.app)
[**→ LinkedIn**](https://www.linkedin.com/in/andrewccross/) |
[**→ Stop TB**](https://www.rtc.stoptb.org/)


---

## Running the demo

**Cloud demo** (Claude only): [link]
Demonstrates RAG pipeline, structured output, cost tracking,
and national guideline upload. Ollama provider not available
in cloud deployment.

**Local deployment** (all providers including Ollama):
Clone the repo and follow the setup instructions to run
with full local model support.

---

## What this does

A health worker describes a person's health case in plain language. The system 
retrieves relevant passages from WHO TB treatment guidelines and generates structured 
clinical guidance showing:

- Recommended action
- Supporting regimen or protocol
- Red flags and referral criteria
- Uncertainty and limitations
- Source citations with page numbers

Every response is grounded in retrieved guideline text. The LLM never 
answers from training data alone.

---

## Why this exists

Digital health tools for LMIC settings face a fundamental tension: 
the best AI models require sending personal data to cloud APIs outside 
national borders, while data sovereignty frameworks increasingly 
require patient data to stay local.

This tool is an illustrative demontration that makes that tradeoff 
explicit and measurable rather than assuming it away. 
Three deployment options are supported:

| Rung | Provider | Data sovereignty | Cost | Quality |
|---|---|---|---|---|
| 1 | Claude / GPT-4o | Data leaves facility | ~$0.01/query | High |
| 4 | Ollama (local) | Data stays on device | $0.00/query | Reduced |

---

## Evaluation results

Tested against 8 ground truth clinical cases derived from WHO TB 
treatment guidelines:

| Dimension | Claude Sonnet | Llama3.2 (Local) | Delta |
|---|---|---|---|
| Overall | 89% | 57% | -32pp |
| Concept coverage | 70% | 35% | -35pp |
| Safety compliance | 96% | 38% | -58pp |
| Cost per session | ~$0.01/query | $0.00 | - |

**Key finding:** Safety compliance — inclusion of specialist referral 
guidance, national guidelines disclaimers, and uncertainty flags — 
is higher for propriertary LLMs and lower for most locally running LLMs.
Simultaneously, costs are obviously higher using third party models. 
For clinical decision support in high-stakes settings, this gap is 
clinically significant, not just a performance metric.

See [`docs/evaluation_methodology.md`](docs/evaluation_methodology.md) 
for full methodology and [`docs/failure_analysis.md`](docs/failure_analysis.md) 
for known limitations.

---

## Features

- RAG pipeline grounded in WHO TB treatment guidelines (7 documents, 
  510 pages, 2,151 chunks)
- Structured clinical output with mandatory uncertainty flagging
- Three LLM providers: Claude (Anthropic), OpenAI, Ollama (local)
- Per-session cost tracking with Ollama $0.00 comparison
- Data sovereignty indicator showing whether data leaves the facility
- National guideline upload — augment WHO base with country-specific 
  PDFs for that session
- Evaluation framework with automated scoring across providers

---

## Architecture
health-worker-copilot/
├── llm/
│   ├── client.py          # Provider abstraction (Claude/OpenAI/Ollama)
│   └── cost_tracker.py    # Token counting and cost estimation
├── rag/
│   ├── ingest.py          # PDF → chunks → embeddings → ChromaDB
│   ├── retriever.py       # Semantic search across guideline chunks
│   └── pipeline.py        # Retrieval + LLM + structured output
├── app/
│   └── main.py            # Streamlit UI
├── eval/
│   ├── test_cases.json    # Ground truth clinical cases
│   └── evaluator.py       # Automated scoring framework
├── knowledge_base/
│   ├── who_tb_guidelines/ # Source PDFs
│   └── chroma_db/         # Pre-built vector store
└── docs/
├── design_decisions.md
├── deployment_ladder.md
├── evaluation_methodology.md
└── failure_notes.md

---

## Deployment ladder

This tool is as a demonstration only, not for programmatic use.
It is designed for programs that need to start with commercial 
APIs and migrate toward local sovereignty as capacity grows.

**Rung 1 → Rung 4 migration requires only a config change:**

```yaml
# config/app_config.yaml
default_provider: ollama  # was: claude
```

See [`docs/deployment_ladder.md`](docs/deployment_ladder.md) for 
full migration guidance and hardware requirements.

---

## Running locally

**Prerequisites:**
- Python 3.10 or 3.11
- [Ollama](https://ollama.com) with `mxbai-embed-large` pulled
- Anthropic API key (for Claude provider)

```bash
git clone https://github.com/andrewccross/health-worker-copilot.git
cd health-worker-copilot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # Add your API key
ollama serve              # In a separate terminal
streamlit run app/main.py
```

The pre-built ChromaDB vector store is included — no re-ingestion 
required.

---

## Knowledge base

7 WHO TB treatment guideline documents, 510 pages, 2,151 chunks.
Embedded locally using `mxbai-embed-large` via Ollama — no data 
sent to external APIs during ingestion.

See [`knowledge_base/README.md`](knowledge_base/README.md) for 
full provenance and update policy.

---

## Design principles

**Tools only I would build.** This is a demonstration of a RAG implementation. 
Design decisions reflect real deployment experience in LMIC 
health systems — from the sovereignty deployment ladder to the 
mandatory uncertainty flagging to the evaluation framework that 
measures safety compliance rather than just accuracy.

**Failure analysis is as important as the demo.** The evaluation 
results show that for now: cost, response time, and guardrails around 
responses for local models is not differ and are explicit. It is what program 
implementers need to know before choosing a deployment rung.

**Open source, documented, reproducible.** Everything needed to 
evaluate, extend, or deploy this tool is in this repository.

---

## Limitations

See [`docs/failure_analysis.md`](docs/failure_analysis.md) for 
full documentation. Key limitations:

- Knowledge base reflects WHO global guidelines; national adaptations 
  may differ
- Evaluation uses keyword-based scoring; paraphrased concepts may 
  score lower than deserved
- Ollama responses take 45-90 seconds on CPU-only hardware
- Cold start latency of ~20 seconds on first browser load

---

## About

Built as an exploratory project to learn/ demonstrate technical credibility at 
the intersection of AI, digital public infrastructure, and global 
health. 

**Andrew Cross** | https://linkedin.com/in/andrewccross