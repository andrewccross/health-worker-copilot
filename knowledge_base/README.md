# Knowledge Base — WHO TB Treatment Guidelines

This directory contains the source documents used to build the RAG 
knowledge base for the Health Worker AI Copilot. All documents are 
official WHO publications downloaded directly from who.int.

The RAG pipeline retrieves relevant chunks from these documents at 
query time. The LLM never answers from training data alone — every 
clinical response is grounded in retrieved guideline text, with 
sources cited in the output.

---

## Source Documents

### Tier 1 — Core Treatment Guidance

| # | Document | Year | URL | Downloaded |
|---|---|---|---|---|
| 1 | WHO Consolidated Guidelines on TB, Module 4: Treatment | 2022 | https://www.who.int/publications/i/item/9789240048126 | July 2026 |
| 2 | WHO Treatment of Drug-Susceptible TB: Quick Reference Guide | 2022 | https://www.who.int/publications/i/item/9789240048164 | July 2026 |
| 3 | WHO TB Treatment Module 4: Operational Handbook | 2022 | https://www.who.int/publications/i/item/9789240048232 | July 2026 |

### Tier 2 — Drug Resistance and HIV Co-infection

| # | Document | Year | URL | Downloaded |
|---|---|---|---|---|
| 4 | WHO Consolidated Guidelines on Drug-Resistant TB Treatment | 2019 | https://www.who.int/publications/i/item/9789241550529 | July 2026 |
| 5 | WHO Consolidated Guidelines on TB, Module 1: Prevention | 2020 | https://www.who.int/publications/i/item/9789241550000 | July 2026 |
| 6 | WHO Guidelines for Treatment of Drug-Susceptible TB (2017 update) | 2017 | https://www.who.int/publications/i/item/9789241550000 | July 2026 |

### Tier 3 — Supplementary

| # | Document | Year | URL | Downloaded |
|---|---|---|---|---|
| 7 | WHO Operational Handbook on TB, Module 2: Screening | 2021 | https://www.who.int/publications/i/item/9789240022676 | July 2026 |
| 8 | WHO Handbook for Digital Technologies to Support TB Medication Adherence | 2017 | https://www.who.int/publications/i/item/9789241514903 | July 2026 |

**Note on document 8:** This handbook was co-developed by the project 
author in collaboration with WHO and the Global Task Force on Digital 
Health for Tuberculosis. It is included both as a knowledge source and 
as evidence of the author's direct contribution to the evidence base 
this tool draws from.

---

## File Naming Convention

Files are named using the pattern:
`WHO_[topic]_[module/type]_[year].pdf`

Example: `WHO_TB_Treatment_Module4_2022.pdf`

---

## Update Policy

WHO TB treatment guidelines are updated on an irregular schedule, 
typically following new evidence reviews. This knowledge base should 
be reviewed when:

- WHO publishes new TB treatment recommendations
- A major drug resistance update is issued
- Country programs flag a discrepancy between this knowledge base 
  and current national guidelines

**Last reviewed:** July 2026
**Next review due:** July 2027
**Reviewed by:** Andrew Cross

---

## Limitations

- This knowledge base reflects WHO global guidelines only. Country 
  programs may follow national adaptations that differ from these 
  recommendations.
- Users can upload country-specific guidelines via the document 
  upload feature. Uploaded documents augment but do not replace 
  the base knowledge base for that session.
- When retrieved chunks from different sources conflict, the system 
  surfaces both and flags the discrepancy. Clinical judgment should 
  resolve conflicts, not this tool.
- Guidelines have a publication date. Emerging resistance patterns 
  or new drug approvals after the document dates may not be 
  reflected.

---

## Embeddings

Documents in this directory are processed by `rag/ingest.py` into 
vector embeddings stored in `chroma_db/`. The ingestion script uses 
`mxbai-embed-large` via Ollama for local embedding generation — 
no data is sent to external APIs during the ingestion process.

To rebuild the knowledge base after adding new documents:

```bash
python rag/ingest.py
```

This is a one-time operation per document set. End users of the 
deployed application do not need to run this.