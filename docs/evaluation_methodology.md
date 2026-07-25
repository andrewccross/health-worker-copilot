# Evaluation Methodology

## Approach

Evaluation uses keyword-based scoring rather than LLM-based evaluation.
This is intentional: keyword scoring is transparent, reproducible, and
auditable — important for a clinical tool where evaluation methodology
will be scrutinized by health system reviewers.

Limitation: keyword matching misses paraphrased concepts. A response
that correctly describes isoniazid as "H" rather than by full name
will score lower than it should. This is documented as a known
limitation, not corrected by softening the methodology.

## Dimensions

Three dimensions scored per test case:

1. Concept coverage: did the response mention expected clinical concepts?
2. Safety compliance: did it include required safety flags?
3. Negative check: did it avoid clinically inappropriate content?

## Results — Claude Sonnet (claude-sonnet-4-6)

Run date: July 2026
Test cases: 8
Knowledge base: WHO TB guidelines, 2151 chunks, 510 pages
Embedding model: sentence-transformers all-MiniLM-L6-v2

| Dimension | Score |
|---|---|
| Overall | 89% |
| Concept coverage | 70% |
| Safety compliance | 96% |
| Total cost (8 cases) | $0.0818 |
| Ollama equivalent | $0.0000 |

## Results — Ollama Llama3.2 (fully local, zero cost)

Run date: July 2026
Hardware: CPU-only laptop (no GPU acceleration)
Embedding model: sentence-transformers all-MiniLM-L6-v2

| Dimension | Score |
|---|---|
| Overall | 57% |
| Concept coverage | 35% |
| Safety compliance | 38% |
| Total cost (8 cases) | $0.0000 |

## Cross-provider comparison

| Dimension | Claude Sonnet | Llama3.2 (Local) | Delta |
|---|---|---|---|
| Overall | 89% | 57% | -32pp |
| Concept coverage | 70% | 35% | -35pp |
| Safety compliance | 96% | 38% | -58pp |
| Cost per 8 cases | $0.0818 | $0.0000 | - |

## Key finding

Safety compliance is the most clinically significant dimension.
Claude included required safety flags in 96% of responses.
Llama3.2 included them in 38% of responses — a 58 percentage
point gap.

For routine first-line queries in well-resourced settings, local
models may be acceptable. For complex cases involving drug
resistance, HIV co-infection, pediatric dosing, or pregnancy,
the safety compliance gap is clinically significant and should
not be dismissed as a performance tradeoff.

## Effect of embedding model change

An earlier evaluation run used Ollama mxbai-embed-large for
embeddings. The current run uses sentence-transformers
all-MiniLM-L6-v2 for cloud deployment compatibility.

Score impact for Claude: -2pp overall, -4pp concept coverage,
-4pp safety compliance. The tradeoff is acceptable given the
benefit of consistent behavior across local and cloud deployment.

See docs/design_decisions.md for full rationale.

## Recommendations for program implementers

Programs considering local deployment for data sovereignty should:

1. Run this evaluation against their own ground truth case mix
   before deployment
2. Supplement local models with mandatory safety disclaimers
   injected at the system prompt level
3. Treat the deployment ladder as a clinical governance decision,
   not a technical one — the choice of model affects patient safety

This tool provides the evaluation infrastructure to make that
decision with evidence rather than assumption.