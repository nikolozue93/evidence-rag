# Evidence Investigation Engine — Evaluation Report

## 1. Scope and evaluation setup

The project is intended to answer questions by finding and comparing evidence across multiple source documents. I evaluated retrieval separately from answer generation because retrieving the correct passage does not guarantee that an LLM uses it correctly.

The main corpus contains **at least 50,000 distinct source passages**, extracted from the HotpotQA *distractor* configuration and indexed in Qdrant. The preparation script removes exact duplicates using a SHA-256 hash of case-folded, whitespace-normalized passage text. The unit counted is a deduplicated source passage, **not a question or an overlapping chunk**. This does not remove semantic near-duplicates. The indexed record count was checked against Qdrant after ingestion.

The application indexes dense MiniLM embeddings (`sentence-transformers/all-minilm-l6-v2`) and sparse BM25 representations (`qdrant/bm25`). Hybrid search combines the ranked candidate lists in Python using weighted reciprocal-rank fusion (0.55 dense, 0.45 sparse; RRF constant 60), with a limit on repeated titles. The fusion logic is our own implementation rather than the database's built-in fusion.

The evaluation questions are drawn from HotpotQA validation examples. Their questions, answers and annotated supporting-source IDs are kept out of the indexed evidence, **but the associated source passages are deliberately indexed**, so the system can search for them. This tests evidence retrieval and answering when relevant material is available; it is not a test of handling entirely unseen documents or organizations.

## 2. Retrieval: dense versus hybrid

Forty questions were evaluated with the same top-eight retrieval limit for each configuration. Both modes completed all 40 questions without errors.

| Metric @8 | Dense only | Hybrid |
|---|---:|---:|
| Supporting-source Recall@8 | 0.7000 | **0.8000** |
| Hit@8 (at least one required source) | 0.9500 | **1.0000** |
| All-supporting-hit@8 (all required sources) | 0.4500 | **0.6000** |
| MRR@8 (first required source) | **0.8604** | 0.8592 |

The most relevant improvement for multi-document questions is complete evidence coverage. Dense retrieval found every required source on **18/40** questions, compared with **24/40** for hybrid retrieval. In contrast, MRR changed very little, indicating that the main benefit was finding *more* relevant sources rather than moving the first relevant result higher.

**Limitation:** Even hybrid retrieval missed at least one required source on **16/40** questions at K=8. This evaluation is relatively small, so the results should not be treated as a guarantee of performance on different datasets.

Raw per-question results: [`retrieval_eval.csv`](retrieval_eval.csv).

## 3. Generated answers and source references

The investigation pipeline uses Groq's `openai/gpt-oss-120b` model. It performs two evidence-search rounds using a structured `search_evidence` tool, then synthesizes a final answer from retrieved passages. If the provider rejects structured tool use, the implementation has a separately recorded application-driven fallback; this fallback must not be reported as an LLM tool call.

A separate **40-question** run evaluated generated answers using the hybrid retrieval configuration:

| Measurement | Observed result | What it measures |
|---|---:|---|
| Completed without recorded errors | **40/40** | Operational completion of this run |
| Expected answer appears literally in generated text | **27/40 (67.5%)** | A string/token proxy, not semantic accuracy |
| All annotated supporting sources retrieved during investigation | **35/40 (87.5%)** | Availability of all gold evidence over the multi-round investigation |
| Answers containing only valid numbered source references, with at least one reference | **39/40 (97.5%)** | Whether cited numbers correspond to returned sources |

The initial answer evaluator recognized `[2]` but not other citation styles emitted by the model, such as `【2】` or `【2†L1-L3】`. As a result, its older CSV reports an incorrectly low citation validity figure (37.5%). An **offline citation-format audit**, performed without additional LLM calls, found valid numbered references in 39/40 saved answers. The application was updated to normalize those styles. The original evaluation CSV is preserved rather than silently rewritten.

**Important limits of these measurements:**

- A literal-match failure does not necessarily mean the answer was wrong. For example, a correct explanation of a yes/no comparison can omit the standalone word `yes` or `no`.
- A numbered citation can refer to a real retrieved source while still failing to support the sentence in which it appears. The **97.5% figure is source-reference validity, not factual citation accuracy or claim-level entailment**.
- The original CSV does not preserve a reliable mapping of all alternative citation-number formats to gold source IDs. Older `all_gold_cited` and `cited_gold_fraction` measurements should therefore **not** be interpreted as corrected gold-source citation scores.
- The 87.5% complete-evidence figure measures the multi-round investigation, not the single top-eight retrieval experiment in section 2. The two results are not directly interchangeable.

Raw answers: [`answer_eval.csv`](answer_eval.csv). Offline audit: [`citation_format_audit.json`](citation_format_audit.json).

## 4. Insufficient and contradictory evidence

Eight small controlled tests were created to check whether the model acknowledges missing facts or conflicting claims:

| Scenario | Cases | Heuristic checks passed |
|---|---:|---:|
| Insufficient information | 4 | **4/4** |
| Contradictory source accounts | 4 | **4/4** |

These cases use actual LLM inference and structured search calls, but feed **fixed synthetic passages directly** instead of testing Qdrant retrieval. The checks look for relevant uncertainty/conflict language and valid citations. A manual reading of the answers is still needed before claiming that every cited claim is correctly supported. The small sample is intended as a behavioral smoke test, not a statistically representative benchmark.

The initial contradiction-check output marked two cases as failures because it did not parse the model's alternative citation format. The saved responses were re-audited, yielding the 4/4 results above. Both raw and audited files are retained.

Raw test responses: [`edge_case_eval_after_fix.csv`](edge_case_eval_after_fix.csv). Audit: [`citation_format_audit.json`](citation_format_audit.json).

## 5. Chunking strategies

The upload pipeline supports fixed-size overlapping word windows and paragraph-aware chunking without overlap. Both were compared descriptively on the same **300-document** sample using a maximum chunk length of 180 words.

| Property | Fixed windows | Paragraph-aware |
|---|---:|---:|
| Documents | 300 | 300 |
| Chunks | 328 | 328 |
| Average chunks/document | 1.093 | 1.093 |
| Mean words/chunk | 88.63 | 86.07 |
| Median words/chunk | 82 | 80 |
| Stored/original word-count ratio | 1.03 | 1.00 |

The fixed-window method repeated about **3%** of the original text because of overlap. The paragraph-aware strategy did not repeat text in this sample. This is an analysis of chunk shape and storage overhead; **neither strategy has been shown to improve retrieval accuracy** in the current experiment. A better follow-up would index identical documents separately under each strategy and compare annotated relevant-chunk recall and answer quality.

Results: [`chunking_comparison.csv`](chunking_comparison.csv).

## 6. Observed failures, limitations, and priorities

1. **Incomplete multi-source retrieval.** Hybrid all-supporting-hit@8 is 60%, so a substantial fraction of questions still lack complete evidence in the initial retrieval result. Better query decomposition, increasing candidate recall before reranking, and tuning the fusion weights are priorities.
2. **Answer scoring uncertainty.** Literal gold-answer inclusion is an imperfect proxy, particularly for paraphrases, yes/no questions and concise reference answers. A human-reviewed sample or a carefully designed semantic evaluation would give a more meaningful estimate.
3. **Citation correctness.** The citation-format bug was corrected, but the current reference-validity audit does not verify claim-level support. Future work should link each claim to the exact supporting passage and test entailment.
4. **Data and domain limitations.** Evaluation is heavily HotpotQA-based rather than drawn from real operations reports. Exact-content deduplication does not detect paraphrased duplicates. Scanned/image-only PDFs are unsupported without OCR.
5. **Operational constraints.** Render's free-tier cold starts, Qdrant Cloud's limited resources, and Groq rate limits may affect latency or availability. The document update workflow uses multiple Qdrant operations rather than a cross-collection transaction, so interrupted writes can require cleanup.

The most useful next improvement would be a larger, manually audited multi-hop and contradiction benchmark with claim-level citation scoring. More UI styling would not address the current quality bottleneck.

## 7. Reproducibility

From the repository root, after configuring environment variables and installing the dependencies:

```bash
python -m scripts.prepare_hotpot --target-records 50000 --eval-questions 60
python -u -m scripts.ingest_hotpot
python -u -m scripts.evaluate_retrieval --limit 40 --k 8
python -m scripts.compare_chunking --max-docs 300
python -u -m scripts.evaluate_answers --limit 40 --max-new 5 --cooldown 2
# Repeat the preceding command in batches to reach 40 questions.
python -u -m scripts.evaluate_edge_cases --out reports/edge_case_eval_after_fix.csv
python -m scripts.audit_citation_formats --answers reports/answer_eval.csv --edges reports/edge_case_eval_after_fix.csv
```

The evaluation outputs are provided in `reports/`. HotpotQA raw/processed records and secrets are excluded from Git. See the repository README and `AI-USAGE.md` for setup and AI-assistance disclosure.
