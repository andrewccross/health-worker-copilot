# Failure Analysis and Known Limitations

## Provider Quality Comparison — Preliminary Observations

### Test case: TB symptom evaluation query
Query: [your test query here]
Date: July 2026

| Dimension | Claude Sonnet | Ollama Llama3.2 |
|---|---|---|
| Output tokens | 526 | 237 |
| Structured sections | Complete | Incomplete |
| Source citation format | Correct page refs | Raw source tags |
| Clinical accuracy | Correct (active TB exclusion) | Correct (general) |
| Specificity | High | Low |
| Speed | ~30s | ~60s+ |
| Cost | $0.0102 | $0.0000 |

### Observed degradation pattern
Ollama produces directionally correct responses but with significantly 
less clinical specificity. The UNCERTAINTY and RED FLAGS sections are 
the most affected — Ollama tends to underspecify conditions requiring 
escalation. For routine first-line queries this may be acceptable. 
For complex cases involving drug resistance or comorbidities, the 
quality gap is clinically significant.

### Retrieval note
Both providers retrieved the same source documents 
(WHO_TB_Prevention_Module1_2020.pdf, WHO_DR_TB_Treatment_2019.pdf),
confirming that retrieval quality is provider-agnostic. The quality 
difference is entirely in the reasoning layer, not the retrieval layer.
This supports the architectural decision to keep retrieval separate 
from generation.

## Known Limitations

### 1. Chunk boundary citations
Page numbers in source citations sometimes reflect the chunk boundary
rather than the exact page containing the recommendation. Citation 
accuracy is approximate, not exact.

### 2. Retrieval misses on explicit regimen queries
Queries asking for specific regimens (e.g. "2HRZE/4HR") sometimes 
retrieve contextual chunks rather than the chunk containing the 
explicit regimen statement. Increasing top_k from 4 to 6 partially 
mitigates this. Full mitigation requires query rewriting.

### 3. Ollama token counting
Ollama does not return reliable token counts. Current implementation
estimates tokens from character count (chars / 4). This is approximate
and will undercount for non-Latin scripts — relevant for multilingual
deployment contexts.

### 4. Response latency
Ollama responses take 45-90 seconds on CPU-only hardware. Unacceptable
for routine clinical use without GPU acceleration. Cloud API providers
respond in 15-30 seconds including retrieval.

### 5. Cold start latency
Streamlit takes 15-20 seconds to initialize on first browser load 
on local hardware. This is framework overhead, not application code.
On Streamlit Community Cloud the cold start is approximately 5-8 
seconds. A production deployment would use a faster framework 
(FastAPI + React) with a proper loading state. Streamlit is 
appropriate for portfolio demonstration purposes.

### 6. Local model safety compliance gap

Evaluation against 8 ground truth cases showed Ollama Llama3.2
achieved 33% safety compliance vs 100% for Claude Sonnet. The local
model frequently omitted specialist referral guidance and national
guidelines disclaimers that the system prompt explicitly requested.

Root cause: smaller local models are less instruction-following than
commercial models, particularly for nuanced safety requirements
embedded in long system prompts.

Partial mitigation: inject mandatory safety disclaimers as a
post-processing step rather than relying on the model to include
them. This is planned for a future release.

Current recommendation: Ollama deployment (Rung 4) is appropriate
for connectivity-constrained environments where cloud APIs are
unavailable, with the understanding that safety compliance is
reduced and additional clinical oversight is required.