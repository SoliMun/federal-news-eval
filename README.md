# Federal News Summarization - Model Evaluation Framework

*This code is an excerpt from a professional project, shared with approval from my employer. Certain details (internal paths, configuration values, and propriety tooling) have been removed or generalized to meet contractual requirements. The core logic, evaluation methodology, and decisions are intact.*

---

## Research Question
Can an off-the-shelf (no fine-tuning) language model reliably generate structured, policy-relevant summaries of federal agency news that meet production quality requirements or does the complexity of the output format (strict JSON schema, bullet constraints, domain-restricted URLs) require some fine-tuning?

This repository implements the evaluation framework used to answer that question. It measures model performance across seven structural dimensions and one content quality metric (ROUGE-L), establishing a quantitative baseline for the base model and a reproducible benchmark for comparing future fine-tuned checkpoints.

## Overview

This package implements a systematic evaluation framework for a fine-tuned language model that generates summaries of federal agency news. The goal is to make sure model outputs meet both structural requirements and content quality standards before delivery to human reviewers. 

The evaluation pipeline runs the model against a reference dataset, applies multi-dimensional quality checks defined in `bullet_rules.py`, computes automated metrics (includeing ROUGE-L), and logs all results to MLflow for reproducible cross-run comparison.

---

## The Problem

A fine-tuned LLM generates daily summaries of federal news articles. Each summary must satisfy strict requirements: correct JSON structure, a specific bullet format, quantitative length constraints, and verified source URLs. Violations at any stage (malformed JSON, missing bullets, hallucinated URLs) would break downstream processing or undermine confidence in the outputs. 

## The Solution

This framework provides a way to measure how well the model satisfies these requirements, enabling data-driven decisions when comparing model versions and tuning hyperparameters. 

---

## Baseline Results

**[`notebooks/baseline_analysis.ipynb`](notebooks/baseline_analysis.ipynb)** contains the full analysis of the base model's performance on a 315-article evaluation set. It walks through the research question, dataset context, pipeline-stage breakdown, gap analysis, and inference latency (with visualizations for each).

**Summary finding:** The base model fails every production threshold, with the largest gaps in output length calibration (word count: −49pp, bullet count: −42pp). The failure pattern confirms that structured output compliance (not content understanding) is the primary bottleneck, and provides the quantitative motivation for fine-tuning.

--

## How it Works

The evaluation pipeline follows these stages:

**Stage 1: Data Loading**
- Load eval dataset (JSONL format)
- Load model, tokenizer, and JSON schema

**Stage 2: Prompt Construction**
- Build system and user message for each article
- Apply chat template formatting

**Stage 3: Model Inference**
- Manage context budget (reserve tokens for output)
- Generate summary with configurable parameters
- Track inference latency

**Stage 4: Output Parsing**
- Extract JSON from model output (tolerant parsing)
- Handle markdown fences, commentary, formatting errors

**Stage 5: Quality Validation**
- Validate JSON structure against schema
- Check bullet formatting and counts
- Verify word count ranges (total and per-bullet)
- Validate source URL domain

**Stage 6: Content Scoring**
- Compute ROUGE-L F1 vs. reference summary
- Record all pass/fail flags

**Stage 7: Aggregation and Logging**
- Calculate pass rates across all articles
- Compute latency percentiles (p50,p95)
- Log parameters, metrics, and artifacts to MLflow

---

## Files

### `bullet_rules.py` (Quality Validation Logic)

Defines the per output validation criteria. Each function checks one dimension of output quality:

|Function | What it checks |
|---------|----------------|
| `bullet_format_ok()` | Every line starts with `- ` |
| `bullet_count_in_range()` | Bullet count is within [3, 5] |
| `total_words_in_range()` | Total word count is within [70, 250] |
| `per_bullet_len_ok()` | Every bullet is within [12, 40] |
| `domain_allowlist_ok()` | Source URL ends in `.gov` or `.mil` |

### `tolerant_json.py` (JSON Parser)

LLMs frequently wrap JSON in markdown fences, prepend commentary, or introduce minor syntax errors like trailing commas. Rather than failing on any malformed output, this module applies a sequence of heuristic cleanups (e.g., stripping fences, collapsing newlines, removing trailing commas) and attempts progressive parsing. This lets the evaluation pipeline recover valid summaries that would otherwise be discarded due to formatting noise. 

### `run_eval.py` (Evaluation Pipeline)

Orchestrates end-to-end pipeline: 

1. **Prompt Construction:** Formats each article into a structured system and user prompt that instructs the model to return strict JSON.
2. **Inference:** Loads the base model (with optional LoRA adapter), manages context budgets, and generates output with configurable decoding parameters.
3. **Tolerant Parsing:** Extracts valid JSON even when the model wraps output in markdown fences or adds extraneous commentary.
4. **Validation:** Applies all checks from `bullet_rules.py`. Includes fallback logic for common failure modes (e.g., list formatted bullets, hallucinated URLs).
5. **Scoring:** Computes ROUGE-L F1 against reference summaries.
6. **Logging:** Aggregates pass rates and latency percentiles across the eval set and logs parameters, metrics, and artifacts to MLflow.

### `notebooks/baseline_analysis.ipynb` (Baseline Analysis)

Exploratory analysis of the base model's performance on the 315-article eval set. Covers research framing, pipeline-stage breakdown, gap analysis, and inference latency. See [Baseline Results](#baseline-results) above.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `json_parse_rate` | % of outputs that are valid JSON |
| `json_schema_rate` | % that match the required output schema |
| `bullet_format_rate` | % with correctly formatted bullet points |
| `bullet_count_in_range_rate` | % with 3-5 bullets |
| `summary_wordcount_rate` | % with total word count in [70, 250] |
| `per_bullet_len_rate` | % with every bullet in [12, 40] words |
| `domain_allowlist_rate` | % with valid `.gov` or `.mil` source URLs |
| `rougeLsum_mean` | Mean ROUGE-L F1 vs. reference summaries |
| `latency_ms_p50` | Median inference latency (ms) |
| `latency_ms_p95` | 95th percentile inference latency (ms) |

--- 

## Quality Thresholds 

These thresholds define the minimum acceptable performance for a model to be considered production ready. They are logged alongside results in MLflow so the pass/fail bar is versioned with each run. 

| Metric | Threshold | Rationale |
|--------|-----------|-------------|
| Schema compliance | >= 95% | Downstream parsing depends on correct structure |
| Bullet format | >= 95% | Reviewers expect consistent formatting |
| Bullet count in [3, 5] | >= 95% | Core output specification |
| Domain allowlist | 100% | Source reliability is non-negotiable for fed data |
| Word count in range | >= 90% | Slightly relaxed; minor length deviations are tolerable |

---

## Key Design Decisions

### 1. Per Bullet Length vs. Average Length 
Validation checks that *every* bullet meets the word count contstraint, not just the average. This prevents a scenario where one very long bullet and one very short bullet average to an acceptable value but produce an inconsistent reading experience.

### 2. URL Fallback Strategy
The model occasionally hallucinates or malforms source URLs. Rather than discarding an otherwise valid summary, the pipeline falls back to the original input URL when the generated URL fails the domain allowlist check. This perserves useful content while maintaining source reliability gaurantees. 

### 3. Bullet Format Coercion
The model sometimes returns bullets as a JSON list rather than a newline separated string, or returns a paragraph without bullet markers. The pipeline normalizes these cases before applying validation. This design choice prioritizes measuring content quality over penalizing minor formatting variations. If the model writes good summaries but formats them as paragraphs, that is a different (and more fixable) problem than if it writes poor content.

### 4. Context Budget Management
Long articles can exceed the model's context window. The pipeline dynamically calculates a context budget based on `max_position_embeddings`, reserving 512 tokens of headroom for generation. This prevents silent truncation errors during inference without hardcoding a fixed input length.

---

## Usage

```bash
# Evaluate base model (no fine-tuning)
python run_eval.py \
  --base_model path/to/base-model \
  --eval_path eval/test_set.jsonl \
  --run_name base-model-eval

# Evaluate fine-tuned model (with LoRA Adapters)
python run_eval.py \
  --base_model path/to/base-model \
  --peft_path path/to/lora-adapter \
  --eval_path path/to/test_set.jsonl \
  --run_name finetuned-v1-eval
```

---

## Input Format

Each line of the eval dataset (`--eval_path`) is a JSON object:

```json
{
  "article_id": "unique-id",
  "agency": "Agency Name",
  "publish_date": "2025-11-24", 
  "title": "Article Title", 
  "url": "https://agency.gov/news/article", 
  "article_body": "Full article text ...",
  "target_summary": "Reference summary used for ROUGE scoring ..."
}
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `transformers` | Model loading and inference (HuggingFace) |
| `peft` | LoRA adapter support |
| `torch` | PyTorch backend |
| `rouge_score` | ROUGE-L metric computation |
| `mlflow` | Experiment tracking and artifact logging |
| `pandas` | Results aggregation |
| `jsonschema` | JSON schema validation |
| `tqdm` | Progress bar for eval loop |
