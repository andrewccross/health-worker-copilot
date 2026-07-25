# Deployment Ladder

## Overview

This tool is designed for health programs that need to start where
the tooling is best and migrate toward local data sovereignty as
capacity and requirements evolve. The ladder has four rungs, each
representing a different balance between clinical quality, data
sovereignty, cost, and infrastructure requirements.

Migration between rungs requires only a configuration change in
`config/app_config.yaml` and a provider selection in the UI.
The RAG pipeline, knowledge base, and evaluation framework are
identical across all rungs.

---

## The Four Rungs

### Rung 1 — Cloud API (default)

```yaml
default_provider: claude
default_model: claude-sonnet-4-6
deployment_rung: cloud_api
```

**How it works:** queries are sent to Anthropic or OpenAI's servers
for inference. Retrieved guideline chunks and the clinical question
leave the facility as part of the API request.

**Appropriate for:**
- Pilots and proof of concept deployments
- High-income country deployments with data sharing agreements
- Organizations with existing cloud data governance frameworks
- Settings where clinical quality is the primary requirement

**Requirements:**
- Internet connectivity
- Anthropic or OpenAI API key
- Budget for token costs (~$0.01-0.02 per query)

**Clinical quality:** highest. Frontier models show 89% overall
evaluation score and 96% safety compliance on WHO TB guidelines
test suite.

**Data sovereignty:** low. Patient case descriptions leave the
facility and are processed on servers outside national jurisdiction.
Identifiers should be stripped before submission (see data
minimization guidance below).

---

### Rung 2 — Cloud API with Data Minimization

```yaml
default_provider: claude
default_model: claude-sonnet-4-6
deployment_rung: cloud_api_minimized
```

**How it works:** same as Rung 1, but clinical queries are
reviewed before submission to ensure no patient identifiers
are included. The tool processes clinical presentations, not
patient records.

**Appropriate for:**
- Organizations with basic data protection requirements
- Programs that need commercial model quality but cannot send
  identified patient data outside national borders

**Requirements:** same as Rung 1, plus a data minimization
protocol for health workers.

**Clinical quality:** identical to Rung 1.

**Data sovereignty:** moderate. Clinical presentation data
leaves the facility but patient identifiers do not.

**Note:** this tool is designed for clinical decision support,
not patient record management. Queries should describe clinical
presentations ("35-year-old, smear positive, no prior treatment")
not patient identities. This is enforced by health worker
training, not technical controls.

---

### Rung 3 — Hosted Open-Weight Model

```yaml
default_provider: ollama
default_model: llama3.1-8b
deployment_rung: hosted_open
```

**How it works:** an open-weight model (Llama 3.1, Mistral,
Phi-3) is hosted on the organization's own cloud infrastructure
or national data center. No data leaves the organization.
Token costs are replaced by compute costs.

**Appropriate for:**
- National programs with cloud infrastructure but strong data
  sovereignty requirements
- Organizations with DevOps capacity to manage model hosting
- Settings where per-query token costs are prohibitive at scale

**Requirements:**
- Cloud infrastructure (AWS, GCP, Azure, or national equivalent)
- DevOps capacity for model deployment and maintenance
- GPU-enabled compute for acceptable inference latency

**Clinical quality:** dependent on model size and hardware.
Llama 3.1 8B significantly outperforms Llama 3.2 2B. Estimated
70-75% overall evaluation score with appropriate hardware.

**Data sovereignty:** high. All data stays within organizational
or national infrastructure.

**Migration from Rung 1:** point the Ollama endpoint at your
hosted model URL instead of localhost. One config change.

---

### Rung 4 — Fully Local Deployment

```yaml
default_provider: ollama
default_model: llama3.2
deployment_rung: fully_local
```

**How it works:** the entire system runs on a local server or
laptop. No internet connection required after initial setup.
The model, knowledge base, and application all run on-device.

**Appropriate for:**
- District health offices with intermittent connectivity
- Air-gapped environments with strict data security requirements
- Programs where zero per-query cost is essential for
  sustainability at scale
- Demonstration of full data sovereignty to government partners

**Requirements:**
- Local hardware (laptop minimum, dedicated server recommended)
- Ollama installed with a compatible model pulled
- No ongoing internet connectivity required

**Clinical quality:** reduced. Llama 3.2 (2B parameters) shows
57% overall evaluation score and 38% safety compliance on the
WHO TB guidelines test suite. See evaluation methodology for
full analysis.

**Data sovereignty:** complete. No data leaves the device at
any stage — not during embedding, retrieval, or inference.

**Latency:** 45-90 seconds per query on CPU-only hardware.
Acceptable for intermittent use, not for high-volume settings.

---

## Choosing the Right Rung

Answer these questions before selecting a deployment rung:

**1. What are your data governance requirements?**
If patient data cannot leave national borders under any
circumstances, start at Rung 3 or 4. If you have existing
cloud data sharing agreements, Rung 1 or 2 is appropriate.

**2. What is your connectivity situation?**
Reliable broadband → any rung. Intermittent connectivity →
Rung 4. Air-gapped → Rung 4 only.

**3. What is your infrastructure capacity?**
No IT team → Rung 1. Basic cloud access → Rung 2 or 3.
DevOps capacity → any rung including hosted open models.

**4. What is your quality threshold?**
High-stakes complex cases (MDR-TB, HIV co-infection) →
Rung 1 or 2. Routine first-line cases → Rung 3 or 4
may be acceptable with appropriate clinical oversight.

**5. What is your cost constraint?**
Token costs prohibitive at scale → Rung 3 or 4.
Pilot or low-volume → Rung 1 or 2.

---

## Migration Guidance

### Rung 1 → Rung 2
No technical changes. Add a data minimization protocol to
your health worker training materials.

### Rung 2 → Rung 3
1. Deploy an open-weight model on your infrastructure
   (see Ollama documentation for server deployment)
2. Update `config/app_config.yaml`:
```yaml
   default_provider: ollama
   default_model: llama3.1-8b
```
3. Point Ollama endpoint at your hosted server URL
4. Re-run evaluation against your ground truth case mix
   to confirm quality meets your threshold

### Rung 3 → Rung 4
1. Install Ollama on local hardware
2. Pull your chosen model: `ollama pull llama3.2`
3. No other changes required

### Any rung → Any rung
The knowledge base, retrieval pipeline, and evaluation
framework are identical across all rungs. Migration is
always reversible.

---

## Data Minimization Guidance

Regardless of deployment rung, health workers should describe
clinical presentations rather than patient identities:

**Appropriate:**
"35-year-old, smear positive, no prior TB treatment,
HIV negative. What regimen?"

**Not appropriate:**
"John Doe, DOB 15/03/1991, MRN 4827364, smear positive..."

This is a training requirement, not a technical control.
The tool does not enforce it programmatically.

---

## Evaluation by Rung

Run the evaluation framework before deploying at any rung
to confirm quality meets your program's threshold:

```bash
python -m eval.evaluator
```

Modify `eval/test_cases.json` to add cases specific to your
context — drug resistance patterns, local comorbidities,
language requirements.

The evaluation results in this repository reflect testing
on a standard laptop with no GPU. Results will differ on
server hardware, with larger models, and in non-English
languages.

---

## Known Limitations by Rung

| Limitation | Rung 1-2 | Rung 3 | Rung 4 |
|---|---|---|---|
| Data leaves facility | Yes | No | No |
| Requires internet | Yes | No | No |
| Token cost | ~$0.01/query | $0 | $0 |
| Safety compliance | 96% | ~75%* | 38% |
| Query latency | 5-15s | 10-30s | 45-90s |
| Multilingual quality | High | Medium | Low |

*Estimated for Llama 3.1 8B. Not yet measured.

See `docs/failure_analysis.md` for full limitation documentation.