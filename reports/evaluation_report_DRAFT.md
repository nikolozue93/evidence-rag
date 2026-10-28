# Evidence Investigation Engine — Evaluation Report (Draft)

**Status:** Complete the remaining measurements after running the scripts. Do not present a draft as final results.

## 1. Dataset and methodology

The corpus was built from HotpotQA `distractor` source passages, deduplicated using SHA-256 of whitespace-normalized, casefolded text. Questions, ground-truth answers and supporting-fact labels were stored separately from Qdrant. The corpus was indexed in Qdrant with dense MiniLM embeddings and BM25 sparse representations. The held-out question set was drawn from the HotpotQA validation split; supporting passages were present in the search corpus so the retrieval task was meaningful.

**Corpus size:** 50,000+ independently deduplicated records indexed, per the successful ingestion verification. Record the exact values from `hotpot_records.manifest.json` and Qdrant's point count in the submitted report.

The retrieval experiment used the same **40 questions** and **K=8** for two configurations: dense search and hybrid search with application-side weighted reciprocal-rank fusion and title diversity. This is a paired comparison on a small sample, not a claim of statistical superiority on unseen domains.

## 2. Retrieval results (measured)

| Metric @8 | Dense | Hybrid |
|---|---:|---:|
| Recall@8 (average supporting-source recall) | 0.7000 | 0.8000 |
| Hit@8 (at least one supporting source) | 0.9500 | 1.0000 |
| All-supporting-hit@8 (every supporting source) | 0.4500 | 0.6000 |
| MRR@8 (first supporting result) | 0.8604 | 0.8592 |

Hybrid retrieval found every required supporting record on **24/40** questions, versus **18/40** for dense retrieval. Its advantage in source coverage is more relevant to multi-hop investigations than the nearly unchanged first-hit rank. However, 16 of 40 hybrid searches still did not retrieve the complete required evidence at K=8.

## 3. Full answer and citation evaluation (add measured results)

Run `scripts/evaluate_answers.py` and report the number of successful questions, errors and actual observed results. The script measures lexical gold-answer inclusion, token overlap, cited-ID validity, the proportion of cited IDs that match annotated supporting sources, and whether all annotated supporting sources were retrieved/cited by the investigation. These are useful **proxies**. Lexical answer inclusion is not equivalent to meaning-level correctness, and citing a correct source ID does not prove that each factual claim is entailed by the cited sentence.

**Measured results:** [Fill only after running. Include examples of failures and manual spot checks.]

## 4. Handling missing and conflicting evidence (add measured results)

Use `scripts/evaluate_edge_cases.py` with four unanswerable and four contradictory synthetic scenarios. The LLM performs genuine structured search calls, but receives fixed preselected fixture passages (no Qdrant retrieval). These tests isolate response behavior: explicit uncertainty vs fabrication; identifying contradictory accounts vs silently choosing one. Phrase-based checks require manual review to verify the model actually cited both conflicting documents and did not invent unsupported facts.

**Measured results:** [Fill after running; give denominator and any failed cases.]

## 5. Chunking comparison (add measured results)

The uploaded-document pipeline supports two approaches: (a) fixed word windows with overlap, and (b) paragraph-aware non-overlapping chunks. `scripts/compare_chunking.py` measures chunk count, mean and median chunk size and redundancy ratio on a fixed sample of source texts. This characterizes indexing overhead and boundary preservation, **not downstream retrieval accuracy**. A full follow-up would index the same corpus with both strategies and compare retrieval against known supporting chunks.

**Measured results:** [Fill from `reports/chunking_comparison.csv`.]

## 6. Limitations and next improvements

The available evaluation set is small and primarily drawn from the same dataset domain; it does not establish performance on arbitrary organizational documents. Exact hashing misses semantically duplicated paraphrases. Evidence-index citations are checked syntactically but not fully verified at the individual claim level. Structured two-round tool calling consumes multiple API calls per question and is sensitive to quota/latency. Qdrant free-tier capacity and Render cold starts affect performance; record any observed failures. PDF text extraction doesn't cover scanned PDFs. The document update workflow spans two Qdrant collections without a distributed transaction, so concurrent updates or failures may leave cleanup work.

## 7. Reproducibility

- `python -m scripts.prepare_hotpot --target-records 50000 --eval-questions 60`
- `python -m scripts.ingest_hotpot`
- `python -m scripts.evaluate_retrieval --limit 40 --k 8`
- `python -m scripts.compare_chunking --max-docs 300`
- `python -m scripts.evaluate_answers --limit 40 --max-new 5` (repeat in batches)
- `python -m scripts.evaluate_edge_cases`

Use the existing `.env.example` and README to explain required variables and the public deployment. The full processed dataset and API credentials should not be committed. Add any observed deployment smoke-test results.

## AI disclosure

AI assistance was used to develop/review implementation ideas, tests and documentation. Replace this with a precise account of which code was AI-assisted and what you personally tested, debugged and verified before submission. Do not assert manual verification of tests you did not run.
