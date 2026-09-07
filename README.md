# Fact Knowledge Layer

A production-grade Python backend (FastAPI) and Next.js frontend prototype that transforms unstructured financial and institutional PDF documents into a structured, queryable layer of **atomic facts** rather than plain text chunks.

---

## 🎯 Core Architectural Goal: The Abstention-Over-Wrong-Answer Principle

In institutional and financial domains, **hallucinating numbers, guessing missing units, or extrapolating ambiguous contexts is catastrophic**. A system that guesses a currency unit or invents a missing fiscal year is actively dangerous for financial analysts and decision-makers.

### Design Principles:
1. **Strict Provenance & Programmatic Block IDs**: The LLM is never allowed to invent page numbers or block IDs. Block IDs (`blk_{doc_id}_p{page}_{idx}`) are deterministically assigned by the Python ingestion engine and merely referenced by the model. The server resolves quotes and bounding boxes back to the source.
2. **Refusal on Ambiguity**: When a table row or paragraph presents a metric without explicit units (e.g., whether a number is in Lakhs, Crores, or Thousands) or without unambiguous temporal bounds, the system **abstains** and records an `EXTRACTION_ABSTAINED` or `MISSING_UNIT` failure in SQLite.
3. **Deterministic Truth Reconciliation**: Candidate pairs are retrieved via semantic similarity, but truth classification (`CORROBORATED`, `CONTRADICTED`, `CONTEXTUAL_DIFFERENCE`, `UNRESOLVED`) follows a strict 6-step deterministic cascade before any LLM adjudication.
4. **Zero-Hallucination Retrieval (RAG)**: Answers to user queries are synthesized exclusively around retrieved verified atomic facts and their cross-document relationships. The model is forbidden from inventing figures not present in the evidence.
5. **No Hardcoded Logic or Schemas**: The ingestion, extraction, reconciliation, and retrieval pipelines generalize to any arbitrary uploaded PDF with zero document-specific schemas, hardcoded filenames, or hardcoded entities.

---

## 🏛️ Demonstration Scenarios & Example Questions

The problem statement requires handling four distinct scenarios across documents. The following curated questions can be tested directly in the chat interface:

### Scenario 1: Corroborated Fact (Cross-Document Agreement)
> **Goal**: Find a fact supported across multiple documents, even if phrased or formatted differently.
- **Example Question A**: *"What is Delhivery's Adjusted EBITDA and does it appear consistently across the annual report and earnings presentation?"*
  - **Expected Outcome**: Identifies consistent EBITDA reporting between `02-delhivery-annual-report-fy24-excerpt.pdf` and `03-delhivery-q4-fy24-earnings-presentation.pdf`. Both source documents and exact pages are cited with a `✓ Corroborated` badge.
- **Example Question B**: *"What is Delhivery's Express Parcel volume or segment performance across documents?"*
  - **Expected Outcome**: Cross-verifies volume metrics reported across disclosure excerpts.

### Scenario 2: Contradiction / Tension Between Documents
> **Goal**: Detect genuine numerical or semantic conflicts where documents disagree on the same metric, entity, and period.
- **Example Question A**: *"What revenue figures does the earnings presentation report across FY2022 and FY2024 — are they consistent?"*
  - **Expected Outcome**: Highlights the variance between periods/filings without guessing or forcing alignment. Tagged with `⚡ Contradiction`.
- **Example Question B**: *"Are there conflicting figures reported for Delhivery's operating revenue between the prospectus and subsequent filings?"*
  - **Expected Outcome**: Highlights divergent figures and presents side-by-side citations for human review.

### Scenario 3: Context-Reconciled Difference (Apparent Conflict Resolved by Context)
> **Goal**: Identify two facts that initially look contradictory but are reconciled once context (e.g., scope, unit, definition, or timeframe) is understood.
- **Example Question A**: *"Delhivery's PIN code reach appears as different numbers across documents — what is the correct figure and how can the discrepancy be explained?"*
  - **Expected Outcome**: Discrepancy (e.g. 4,445 vs 700) is reconciled by discovering that one figure measures *total active PIN codes nationwide*, whereas the other describes *incremental pin codes added in tier-2/tier-3 hubs*. Tagged with `⧗ Context-Resolved`.
- **Example Question B**: *"Why do Delhivery's reported financial numbers differ between standalone and consolidated statements?"*
  - **Expected Outcome**: Explains how corporate entity scope (`consolidated` vs `standalone`) reconciles the differing numbers.

### Scenario 4: Abstention Over Hallucination (Refusal to Guess)
> **Goal**: System must explicitly refuse to answer when the source document lacks sufficient detail, units, or evidence.
- **Example Question A**: *"What is the exact unit and measurement basis for macroeconomic figures in the Economic Survey?"*
  - **Expected Outcome**: The system explicitly refuses to guess unstated units (e.g., whether figures are in Crores, Millions, or percentages) and cites logged extraction abstentions (`EXTRACTION_ABSTAINED`).
- **Example Question B**: *"What was Delhivery's projected international revenue for 2030?"*
  - **Expected Outcome**: Returns a direct abstention stating that no supporting atomic fact exists in the knowledge base, refusing to invent figures.

---

## ☁️ Deployment Guide

The application is architected as a **decoupled Full-Stack system**:
- **Frontend**: Next.js (can be deployed on **Vercel**)
- **Backend**: FastAPI + SQLite + PyMuPDF (deploy on **Render**, **Railway**, or **Fly.io**)

### Part 1: Deploying the Backend (Render / Railway)

Because the Python backend requires a long-running server with file/PDF processing libraries (`PyMuPDF`, `pdfplumber`, `docling`), deploy it to a container/Python cloud host:

#### Option A: Deploy on Render (Recommended & Free Tier Available)
1. Go to [Render.com](https://render.com) and click **New + Web Service**.
2. Connect your GitHub repository: `https://github.com/JKSANJAY27/Fact-Knowledge-Layer`.
3. In the setup form:
   - **Name**: `fact-knowledge-layer-api`
   - **Environment**: `Docker` (Render will automatically detect the provided `Dockerfile`)
   - **Instance Type**: Free or Starter
4. Add **Environment Variables**:
   - `GEMINI_API_KEY`: Your Gemini API key
   - `GEMINI_MODEL`: `gemini-flash-latest` (or `gemini-1.5-flash`)
   - `PORT`: `8000`
5. Click **Deploy Web Service**.
6. Note down your backend URL (e.g., `https://fact-knowledge-layer-api.onrender.com`).

#### Option B: Deploy on Railway
1. Go to [Railway.app](https://railway.app) and select **Deploy from GitHub repo**.
2. Select `Fact-Knowledge-Layer`. Railway uses the repository `Dockerfile` automatically.
3. In **Variables**, add `GEMINI_API_KEY` and `GEMINI_MODEL`.
4. Generate a public domain under service settings (e.g., `https://...up.railway.app`).

---

### Part 2: Connecting the Vercel Frontend to the Deployed Backend

Now that your frontend is deployed on Vercel:

1. Open your Vercel project dashboard.
2. Navigate to **Settings** > **Environment Variables**.
3. Add a new variable:
   - **Key**: `NEXT_PUBLIC_BACKEND_URL`
   - **Value**: Your deployed backend URL (e.g., `https://fact-knowledge-layer-api.onrender.com` without a trailing slash).
4. Go to **Deployments** and click **Redeploy** on the latest deployment so Next.js picks up the new environment variable.

Now, all `/api/*` and `/static/*` requests (including chat, uploads, and PDF views) made by the Vercel app will seamlessly proxy to your live backend!

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
│  FastAPI Backend    │   │   Next.js Chatbot UI  │
│ (REST API & RAG Q&A)│   │ (Split PDF Viewer)    │
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
Candidate fact pairs are filtered by embedding/token similarity across different documents, then evaluated through a strict deterministic cascade:
1. **Entity Match**: Do the entities match?
2. **Metric Match**: Do the metrics represent the same financial/operational concept?
3. **Unit Compatibility**: Are units convertible?
4. **Period Check**: Do reporting periods match? If not $\rightarrow$ `CONTEXTUAL_DIFFERENCE` (`difference_field="period"`).
5. **Scope Check**: Do scopes match (Consolidated vs Standalone)? If not $\rightarrow$ `CONTEXTUAL_DIFFERENCE` (`difference_field="scope"`).
6. **Value Equality Check**:
   - If within $1\%$ numerical tolerance $\rightarrow$ `CORROBORATED`.
   - If values conflict $\rightarrow$ `CONTRADICTED` (labeled explicitly as *"likely contradiction"*, never certain).
7. Genuinely ambiguous pairs are routed to structured LLM adjudication or marked `UNRESOLVED`.

### 5. Grounded Query & PDF Citation Interface (`fact_layer/retrieval.py` + `frontend/app/page.js`)
- Hybrid retrieval combines structured SQL filtering (by entity, period, scope) and token overlap.
- Zero-hallucination guarantee: Answers are synthesized strictly from retrieved facts, citing `[Fact #]` and document page numbers.
- **Embedded PDF Viewer**: Clicking any citation card in the chat response instantly opens the source PDF directly at the exact cited page (`#page=<num>`), allowing instant human verification of the claims.
- **Arbitrary PDF Upload**: Users can drag & drop any external PDF in the sidebar; it is ingested, extracted, and reconciled asynchronously, after which it becomes immediately queryable.

---

## 📁 Repository Structure

```
├── Dockerfile              # Docker container definition for backend deployment
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
├── frontend/               # Next.js 14 Interactive Chatbot & PDF Viewer (Vercel-ready)
│   ├── app/
│   │   ├── layout.js
│   │   ├── page.js         # Conversational UI with side-by-side PDF citation viewer
│   │   └── globals.css     # Dark-mode financial terminal design system
│   ├── next.config.js      # API proxy rewrite configuration
│   └── package.json
├── starter-datasets/       # Curated financial & institutional PDFs
│   ├── delhivery/          # Prospectus, Annual Report FY24, Q4 Earnings Presentation
│   └── india-macroeconomy/ # Economic Survey 2024-25, RBI Annual Report, IMF Article IV
├── scripts/
│   ├── ingest_starter_data.py # Batch ingestion script for starter datasets
│   ├── deduplicate_facts.py   # Content-hash fact and relationship deduplication
│   ├── run_llm_extraction.py  # LLM atomic claim extraction runner
│   └── test_four_cases.py     # Verification script for the 4 core cases
├── tests/
│   └── test_pipeline.py    # Unit & integration test suite (pytest)
├── pytest.ini              # Pytest configuration
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🚀 Local Development Quickstart

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
pytest
```

### 4. Run the FastAPI Backend
```bash
uvicorn fact_layer.api:app --host 127.0.0.1 --port 8000
```
Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`.

### 5. Launch the Next.js Frontend
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000` to interact with the Next.js interface.

---

## 📊 Evaluation & Verification Summary

- **Automated Test Suite**: 6 tests covering numeric parsing, unit conversion, temporal equivalence (`FY2024` = `FY 2023-24` = `Year ended March 31 2024`), the 6-step deterministic reconciliation cascade, and zero-hallucination abstention.
- **Dataset Ingestion**: Successfully indexed 6 institutional documents across 101 pages, 3,078 layout blocks, and cross-document relationships.
- **Remote Repository**: Pushed and synced with `https://github.com/JKSANJAY27/Fact-Knowledge-Layer.git`.
