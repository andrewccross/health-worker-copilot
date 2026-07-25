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

| Dimension | Score |
|---|---|
| Overall | 91% |
| Concept coverage | 74% |
| Safety compliance | 100% |
| Total cost (8 cases) | $0.0843 |
| Ollama equivalent | $0.0000 |

## Interpretation

Safety compliance is perfect — every response included appropriate
uncertainty flagging, specialist referral guidance, and national
guidelines disclaimers. For a clinical decision support tool, this
is the most critical dimension.

Concept coverage gap (74%) reflects a known RAG retrieval limitation:
the correct guideline chunks exist in the knowledge base but
occasionally rank outside the top_k=4 retrieved results. Increasing
top_k to 6 partially mitigates this at the cost of longer context
windows and higher token usage.

## Next steps for evaluation

1. Run same test suite against Ollama (llama3.2) to generate
   comparison table
2. Run against OpenAI GPT-4o when API key available
3. Add multilingual test cases (French, Vietnamese) to evaluate
   language degradation on local models
4. Increase test suite from 8 to 30+ cases before publication


==================================================
EVALUATION SUMMARY
==================================================
Cases run:           8
Successful:          8
Average overall:     56%
Concept coverage:    34%
Safety compliance:   33%
Total cost:          $0.0000
Ollama equivalent:   $0.0000