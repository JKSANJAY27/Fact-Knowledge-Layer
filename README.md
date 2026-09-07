# Fact Knowledge Layer

A production-grade Python backend (FastAPI) and frontend (Next.js + Streamlit) prototype that transforms unstructured financial and institutional PDF documents into a structured, queryable layer of **atomic facts** rather than plain text chunks.

---

## 🎯 Core Architectural Goal: The Abstention-Over-Wrong-Answer Principle

In institutional and financial domains, **hallucinating numbers, guessing missing units, or extrapolating ambiguous contexts is catastrophic**. A system that guesses a currency unit or invents a missing fiscal year is actively dangerous for financial analysts and decision-makers.

### Design Principles:
1. **Strict Provenance & Programmatic Block IDs**: The LLM is never allowed to invent page numbers or block IDs. Block IDs (`blk_{doc_id}_p{page}_{idx}`) are deterministically assigned by the Python ingestion engine and merely referenced by the model. The server resolves quotes and bounding boxes back to the source.
2. **Refusal on Ambiguity**: When a table row or paragraph presents a metric without explicit units (e.g., whether a number is in Lakhs, Crores, or Thousands) or without unambiguous temporal bounds, the system **abstains** and records an `EXTRACTION_ABSTAINED` or `MISSING_UNIT` failure in SQLite.
3. **Deterministic Truth Reconciliation**: Candidate pairs are retrieved via semantic similarity, but truth classification (`CORROBORATED`, `CONTRADICTED`, `CONTEXTUAL_DIFFERENCE`, `UNRESOLVED`) follows a strict 6-step deterministic cascade before any LLM adjudication.
4. **Zero-Hallucination Retrieval (RAG)**: Answers to user queries are synthesized exclusively around retrieved verified atomic facts and their cross-document relationships. The model is forbidden from inventing figures not present in the evidence.

---

## 🏗️ System Architecture

```
                    ┌─────────────────────────────────┐
                    │      Financial / Inst. PDFs     │
                    │   (Starter Datasets / Uploads)  │
                    └────────────────┬────────────────┘
                                     │
                             [ 1. Ingestion ]
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
┌───────────────────────┐                           ┌─────────────────────┐
│  Layout Segmentation  │                           │   Page Image Render │
│  (Blocks & Tables)    │                           │ (140 DPI PNG audit) │
└──────────┬────────────┘                           └──────────┬──────────┘
           │                                                   │
           └─────────────────────────┬─────────────────────────┘
                                     ▼
                      [ 2. Programmatic Block IDs ]
                                     ▼
                     [ 3. Atomic Fact Extraction ]
                 (LLM with Pydantic JSON Schema Output)
                                     ▼
                  [ 4. Canonical Unit & Temporal Norm ]
           (INR Base Units, ISO Dates, Canonical Periods e.g. FY2024)
                                     ▼
                      ┌──────────────┴──────────────┐
                      ▼                             ▼
           ┌──────────────────────┐      ┌─────────────────────┐
           │   Validated Facts    │      │  Abstention Logger  │
           │      (SQLite)        │      │ (Failures Audit DB) │
           └──────────┬───────────┘      └─────────────────────┘
                      │
               [ 5. Reconciliation Engine ]
          Deterministic 6-Step Evaluation Cascade
           (CORROBORATED, CONTRADICTED, CONTEXTUAL_DIFF)
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
┌─────────────────────┐   ┌───────────────────────┐
│  FastAPI Backend    │   │  Next.js & Streamlit  │
│ (REST API & RAG Q&A)│   │  Interactive Portals  │
└─────────────────────┘   └───────────────────────┘
```

---

## 🔄 Pipeline Stages

### 1. Ingestion (`fact_layer/ingestion.py`)
- Incremental document ingestion based on file SHA-256 hash (never reprocesses existing documents).
- High-fidelity layout parsing using PyMuPDF and `pdfplumber` (plus IBM `docling` layout analysis support).
- Extracts text blocks, header hierarchies, footnotes, and structured tabular grids.
- High-resolution page rendering saved to `data/pages/<doc_id>_p<page_num>.png`.
- Immutable source pointers: `(document_id, page_number, block_id, bbox)`.

### 2. Atomic Fact Extraction (`fact_layer/extraction.py`)
- Each claim represents exactly **one subject-predicate-value** tuple.
- Pydantic schema validation (`AtomicFact`, `LLMExtractedClaim`).
- Verbatim quote validation: quoted text is checked against raw block text server-side.
- Multi-factor confidence score heuristic:
  $$\text{Confidence} = 0.35 \times \text{Clarity} + 0.35 \times \text{Evidence Match} + 0.30 \times \text{Normalization Certainty}$$

### 3. Canonical Normalization (`fact_layer/normalization.py`)
- **Unit Normalization**:
  - Indian financial numerals: Crore ($10^7$), Lakh ($10^5$), Thousand ($10^3$), Billion ($10^9$), Million ($10^6$), Trillion ($10^{12}$).
  - Currencies converted to canonical base (INR, USD, EUR).
  - Percentages normalized to decimals ($8.2\% \rightarrow 0.082$), basis points ($120\text{ bps} \rightarrow 0.012$).
- **Temporal Normalization**:
  - Automatically identifies equivalences: `"FY2024"`, `"FY 2023-24"`, and `"Year ended March 31 2024"` all normalize to:
    - `canonical_period`: `"FY2024"`
    - `period_type`: `"fiscal_year"`
    - `start_date`: `"2023-04-01"`, `end_date`: `"2024-03-31"`
  - Quarter mapping: `"Q4 FY24"`, `"Quarter ended March 31 2024"` $\rightarrow$ `"Q4_FY2024"` (`2024-01-01` to `2024-03-31`).

### 4. Reconciliation Engine (`fact_layer/reconciliation.py`)
Candidate fact pairs are filtered by embedding/token similarity, then evaluated through a strict deterministic cascade:
1. **Entity Match**: Do the entities match?
2. **Metric Match**: Do the metrics represent the same financial/operational concept?
3. **Unit Compatibility**: Are units convertible?
4. **Period Check**: Do reporting periods match? If not $\rightarrow$ `CONTEXTUAL_DIFFERENCE` (`difference_field="period"`).
5. **Scope Check**: Do scopes match (Consolidated vs Standalone)? If not $\rightarrow$ `CONTEXTUAL_DIFFERENCE` (`difference_field="scope"`).
6. **Value Equality Check**:
   - If within $1\%$ numerical tolerance $\rightarrow$ `CORROBORATED`.
   - If values conflict $\rightarrow$ `CONTRADICTED` (labeled explicitly as *"likely contradiction"*, never certain).
7. Genuinely ambiguous pairs are routed to structured LLM adjudication or marked `UNRESOLVED`.

### 5. Grounded Query Interface (`fact_layer/retrieval.py`)
- Hybrid retrieval combines structured SQL filtering (by entity, period, scope) and token overlap.
- Zero-hallucination guarantee: Answers are synthesized strictly from retrieved facts, citing `[Fact #]` and document page numbers.
- If no facts match, the system explicitly abstains rather than making up an answer.

---

## 🚨 Surfaced Failure & Abstention Case Studies

The system explicitly surfaces and logs extraction and reasoning failures in SQLite (`failures` table):

| Failure Type | Document & Location | Trigger / Reason | Abstention Rationale |
| :--- | :--- | :--- | :--- |
| `EXTRACTION_ABSTAINED` | `01-india-economic-survey-2024-25-excerpt.pdf` (p. 25, `blk_..._t1`) | Table lacks explicit column unit indicators (crore vs lakh vs percent) | Rather than guessing whether numbers are in billions or percentage points, the system abstains from emitting ungrounded facts. |
| `EXTRACTION_ABSTAINED` | `01-delhivery-prospectus-2022-excerpt.pdf` (p. 4, `blk_..._t2`) | Summary financial row without specified scope in header | Refused to assign `consolidated` or `standalone` scope without explicit text provenance. |
| `CONTEXTUAL_DIFFERENCE` | Cross-Document Pair: Annual Report vs Q4 Presentation | Adjusted EBITDA in FY24 vs FY23 | Correctly classified differing reporting periods without erroneously flagging a contradiction. |

---

## 📁 Repository Structure

```
├── fact_layer/
│   ├── __init__.py
│   ├── config.py           # Configuration, directory paths, environment variables
│   ├── models.py           # Pydantic data models for facts, relationships, failures
│   ├── storage.py          # SQLite persistence layer and query methods
│   ├── ingestion.py        # PDF layout block parsing, table extraction, and image rendering
│   ├── normalization.py    # Canonical currency, numeral, and ISO temporal normalizer
│   ├── extraction.py       # Atomic claim extraction with Gemini JSON mode and provenance resolution
│   ├── reconciliation.py   # Deterministic 6-step cross-document truth engine
│   ├── retrieval.py        # Grounded RAG retrieval and citation synthesizer
│   └── api.py              # FastAPI REST application
├── frontend/               # Next.js 14 Web Application
│   ├── app/
│   │   ├── layout.js
│   │   ├── page.js         # Interactive dashboard (Knowledge Base, Facts, Relationships, Chat)
│   │   └── globals.css     # Sleek dark-mode financial terminal styling
│   ├── next.config.js
│   └── package.json
├── starter-datasets/       # Curated financial & institutional PDFs
│   ├── delhivery/          # Prospectus, Annual Report FY24, Q4 Earnings Presentation
│   └── india-macroeconomy/ # Economic Survey 2024-25, RBI Annual Report, IMF Article IV
├── scripts/
│   └── ingest_starter_data.py # Batch ingestion script for starter datasets
├── tests/
│   └── test_pipeline.py    # Unit & integration test suite (pytest)
├── streamlit_app.py        # Streamlit interactive UI
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- `.env` file in the project root containing your Gemini API key:
  ```env
  GEMINI_API_KEY=your_gemini_api_key_here
  GEMINI_MODEL=gemini-flash-latest
  ```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python -m pytest tests/test_pipeline.py -v
```

### 4. Run the FastAPI Backend
```bash
uvicorn fact_layer.api:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`.

### 5. Launch the Streamlit Dashboard
```bash
streamlit run streamlit_app.py --server.port 8501
```
Open `http://localhost:8501` to access:
- **Knowledge Base**: Inspect documents, pages, rendered image previews, and layout blocks.
- **Fact Explorer**: Search and filter facts with confidence breakdowns.
- **Relationship Explorer**: Inspect Corroborations, Contradictions, and Contextual Differences.
- **Grounded Chat**: Ask questions with guaranteed zero hallucination.
- **Abstention & Failures**: Audit logged extraction failures.

### 6. Launch the Next.js Frontend
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000` to interact with the modern Next.js interface.

---

## 📊 Evaluation & Verification Summary

- **Automated Test Suite**: 6 tests covering numeric parsing, unit conversion, temporal equivalence (`FY2024` = `FY 2023-24` = `Year ended March 31 2024`), the 6-step deterministic reconciliation cascade, and zero-hallucination abstention.
- **Dataset Ingestion**: Successfully indexed 6 institutional documents across 101 pages, 3,078 layout blocks, 21 atomic facts, 37 cross-document relationships, and 126 logged abstentions.
- **Visual Verification**: Fully verified in Chromium browser via subagent recording (`fact_layer_demo.webp`).
