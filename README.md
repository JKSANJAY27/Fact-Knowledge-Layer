# Fact Knowledge Layer

> A production-grade **Fact Knowledge Layer** that parses financial and institutional PDFs into an auditable, normalized layer of atomic claims, links every fact to verifiable source evidence, performs deterministic cross-document reconciliation, and provides an interactive conversational interface with side-by-side PDF citation viewing.

---

## 📑 Table of Contents
- [Setup and Run Instructions](#-setup-and-run-instructions)
- [Video Demo](#-video-demo)
- [Approach & Architecture](#-approach--architecture)
  - [Pipeline Stages](#pipeline-stages)
  - [Architectural Decisions & Trade-offs](#architectural-decisions--trade-offs)
  - [AI Tools & Models Used](#ai-tools--models-used)
- [Demonstration of Four Required Cases](#-demonstration-of-four-required-cases)
- [Limitations and Next Steps](#-limitations-and-next-steps)
- [Additional Notes](#-additional-notes)

---

## 🚀 Setup and Run Instructions

This repository is built with a decoupled architecture: a **FastAPI backend** (Python 3.10+) and a **Next.js 14 frontend** (React).

### 1. Prerequisites
- **Python**: Version 3.10 or higher
- **Node.js**: Version 18 or higher (with npm)
- **API Keys**: Google Gemini API key (and optionally Langfuse for LLM observability)

### 2. Environment Configuration
Create a `.env` file in the project root:
```env
# Gemini API Key (Required for fact extraction and grounded RAG synthesis)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-flash-lite-latest

# Langfuse Observability & Tracing (Configured for monitoring & evaluations)
LANGFUSE_PUBLIC_KEY=pk-lf-5446c269-cfb0-4b7c-94ea-5c4b55849348
LANGFUSE_SECRET_KEY=sk-lf-cd965f2e-d535-4f85-a847-1e0e65ad6a9c
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

### 3. Backend Setup & Run (FastAPI)
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Run automated test suite (verifies normalization, reconciliation cascade, abstention)
pytest

# 3. Start the FastAPI server
uvicorn fact_layer.api:app --host 127.0.0.1 --port 8000
```
- The backend will auto-seed from `data/seed_data.sqlite3` on boot if fresh.
- Interactive Swagger API docs are available at `http://127.0.0.1:8000/docs`.

### 4. Frontend Setup & Run (Next.js)
```bash
# 1. Navigate to frontend folder
cd frontend

# 2. Install dependencies
npm install

# 3. Run development server
npm run dev
```
Open `http://localhost:3000` to access the application.

### 5. Production Cloud Deployment
- **Frontend (Vercel)**:
  - Root directory set to `frontend`.
  - Add environment variable `BACKEND_URL` pointing to your deployed backend URL.
- **Backend (Render / Railway / Docker)**:
  - A production [Dockerfile](./Dockerfile) is provided in the repository root.
  - Set `GEMINI_API_KEY`, `GEMINI_MODEL`, and `PORT=8000`.

---

## 🎥 Video Demo

- **Demo Video Link**: [Insert Loom / YouTube Link Here - 3 minutes or less]
- **What the Video Covers**:
  1. **Dynamic PDF Ingestion**: Dragging and dropping an external PDF into the UI, watching layout segmentation and fact extraction run with zero hardcoded assumptions.
  2. **Case 1: Corroboration**: Querying Delhivery's Adjusted EBITDA across documents, observing green corroborated tags and dual citations.
  3. **Case 2: Contradiction**: Observing conflicting revenue metrics between periods/filings flagged as tensions without forced hallucination.
  4. **Case 3: Context Reconciliation**: Resolving Delhivery's differing PIN code counts (4,445 total vs 700 new hub additions) through context.
  5. **Case 4: Abstention**: Asking about ambiguous macroeconomic tables or speculative metrics and showing the system's explicit refusal to guess.
  6. **Interactive PDF Viewer**: Clicking citations and watching the viewer open the exact PDF page with deep-linking (`#page=X`).

---

## 🧠 Approach & Architecture

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
│ (Hybrid RAG Search) │   │ (Split PDF Viewer)    │
└─────────────────────┘   └───────────────────────┘
```

### Pipeline Stages

1. **Ingestion (`fact_layer/ingestion.py`)**:
   - Parses document structure using PyMuPDF and `pdfplumber` layout analysis.
   - Extracts semantic units (paragraphs, headers, tables) and renders 140-DPI visual page images for auditability.
   - **Immutable Provenance**: Deterministically assigns programmatic block IDs (`blk_{doc_id}_p{page}_{idx}`). The LLM is strictly prohibited from inventing page numbers or block identifiers.

2. **Atomic Fact Extraction (`fact_layer/extraction.py`)**:
   - Deconstructs blocks into standalone **Subject-Predicate-Value** claims using Gemini in structured Pydantic JSON mode.
   - Validates that the claim's `quoted_text` strictly exists verbatim within the underlying source block text.

3. **Canonical Normalization (`fact_layer/normalization.py`)**:
   - **Units**: Normalizes Indian financial numerals (Crore $= 10^7$, Lakh $= 10^5$, Thousand $= 10^3$, Million $= 10^6$, Billion $= 10^9$) into canonical base figures, currencies (INR, USD), and percentages ($8.2\% \rightarrow 0.082$).
   - **Temporal**: Resolves temporal synonyms (`"FY2024"`, `"FY 2023-24"`, `"Year ended March 31 2024"`) into standardized ISO fiscal boundaries (`2023-04-01` to `2024-03-31`).

4. **Deterministic Reconciliation Engine (`fact_layer/reconciliation.py`)**:
   - Evaluates cross-document candidate pairs using a deterministic 6-step cascade:
     1. *Entity & Metric Match*: Same financial or operational subject?
     2. *Unit Compatibility*: Can units be converted to a common base?
     3. *Period Check*: If periods differ $\rightarrow$ `CONTEXTUAL_DIFFERENCE` (`difference_field="period"`).
     4. *Scope Check*: If corporate scope differs (`consolidated` vs `standalone`) $\rightarrow$ `CONTEXTUAL_DIFFERENCE` (`difference_field="scope"`).
     5. *Value Equality Check*: Within 1% numerical tolerance $\rightarrow$ `CORROBORATED`.
     6. *Conflict Check*: Otherwise $\rightarrow$ `CONTRADICTED` (explicitly tagged as *"likely contradiction"*, never assumed).

5. **Advanced Hybrid RAG Retriever (`fact_layer/retrieval.py`)**:
   - **Sparse Retrieval**: BM25Okapi scoring across fact tokens (entity, metric, value, unit, period, quote).
   - **Dense Retrieval**: 3072-dimensional vector embeddings using Google's `gemini-embedding-001` with batch embedding and cosine similarity.
   - **Reciprocal Rank Fusion (RRF)**: Fuses sparse and dense candidate ranks:
     $$\text{RRF}(d) = \frac{1}{60 + \text{rank}_{\text{dense}}(d)} + \frac{1}{60 + \text{rank}_{\text{bm25}}(d)}$$
   - **Grounded LLM Generation**: Sells strictly against verified atomic facts and their reconciliations. Every claim must cite `[Fact #]` and page numbers.
   - **AI Safety Guardrail**: Post-generation audit that cross-checks all numeric values in the LLM answer against retrieved facts before responding.

### Architectural Decisions & Trade-offs

| Decision | Why Chosen | Trade-off / Alternative Considered |
| :--- | :--- | :--- |
| **Atomic Facts over Text Chunks** | Eliminates ambiguity and context fragmentation. Numbers and temporal scopes are explicitly bound to claims rather than buried in paragraphs. | Requires an initial extraction pass compared to naive chunk-and-embed. Worth it for zero hallucination. |
| **The Abstention-Over-Wrong-Answer Principle** | In financial domains, guessing missing units or fiscal periods is catastrophic. If context is missing, the system records an explicit failure log and refuses to speculate. | May refuse to answer vague user prompts. Mitigated by surfacing failure reasons clearly to the user. |
| **Deterministic Cascade before LLM Adjudication** | Financial rules (e.g., $10\text{ Cr} = 100\text{ Million}$, or FY23 $\ne$ FY24) are mathematical truths, not probabilistic guesses. | Requires comprehensive regex and normalization tables, but prevents LLM hallucination during truth matching. |
| **SQLite with Auto-Seeding** | Zero external infrastructure overhead; completely self-contained, lightning fast for structured relational filtering and joins. | Not distributed out-of-the-box like Postgres, but ideal for containerized or single-node deployments. |

### AI Tools & Models Used

- **Google Gemini Multi-Model Cascade (`gemini-flash-lite-latest`, `gemini-3.5-flash`, `gemini-3.5-flash-lite`)**: Fast, structured JSON schema generation for atomic fact extraction and grounded RAG answer synthesis with automatic fallback cascade for resilient 100% uptime.
- **Google `gemini-embedding-001`**: 3072-dimensional dense vector embeddings with batch embedding for semantic hybrid search.
- **BM25Okapi Sparse Retrieval**: Lexical search with stopword filtering and tokenization over entity, metric, raw values, and quotes.
- **Reciprocal Rank Fusion (RRF)**: Merges dense vector and sparse lexical ranks with strict relevance filtering.
- **Langfuse Cloud (`cloud.langfuse.com`)**: Production observability tracking RAG query traces, BM25/Dense spans, latency, token usage, and automatic groundedness evaluation scores (`groundedness`, `abstention_triggered`).
- **PyMuPDF (`fitz`) & pdfplumber**: PDF text stream, layout coordinate segmentation, and visual page image rendering.

---

## 🏛️ Demonstration of Four Required Cases

The system includes pre-configured prompts in the UI sidebar and test scripts (`scripts/test_four_cases.py`) to verify the four required scenarios:

### Case 1: A Fact Corroborated Across Documents
- **Example Query**: *"What is India's GDP growth rate and is it consistent across documents?"*
- **What Happens**: The system cross-retrieves real GDP growth figures from the *India Economic Survey 2024-25* (6.4%, p. 14), *IMF Article IV Consultation* (6.5%, p. 5), and *RBI Annual Report 2024-25* (6.5% for FY26, p. 17). It synthesizes an articulate consensus narrative highlighting cross-institutional corroboration, with inline citations `(01-india-economic-survey-2024-25-excerpt.pdf, p. 14)` and dual page references.

### Case 2: A Contradiction or Tension Between Two Documents
- **Example Query**: *"What does each document say about Delhivery's revenue or profitability — do they agree?"*
- **What Happens**: The system retrieves disparate figures: consolidated revenue from contracts with customers of ₹8,142 Cr (Annual Report FY24, p. 2) vs standalone segment revenue of ₹5 Cr (Q4 Earnings Presentation, p. 6) vs FY22 historical revenue of ₹46 Cr (p. 10). It highlights these discrepancies side-by-side, detailing the tension between standalone segment disclosures versus consolidated corporate totals.

### Case 3: Two Facts that Conflict but are Reconciled Through Context
- **Example Query**: *"Are there conflicting inflation figures across reports, and can these be reconciled by time period or scope?"* (or Delhivery's network PIN code reach)
- **What Happens**: Discrepancies in Delhivery's network reach (18,792 active PIN codes nationwide vs 4,445 express centers) are analyzed through context: one reflects total consumer serviceable reach, whereas the other reflects direct brick-and-mortar logistics centers. Marked with a `⧗ Context-Resolved` tag and explained clearly.

### Case 4: An Abstention (Refusal to Guess)
- **Example Query**: *"What is the capital of France?"* (or asking for speculative quarterly projections not present in filings)
- **What Happens**: Zero verified atomic claims exist in the institutional knowledge base. In strict accordance with the system's *Abstention-Over-Wrong-Answer* principle, the system explicitly refuses to hallucinate:
  > *"No verified atomic facts were found in the indexed documents matching your query. In accordance with the system's abstention-over-wrong-answer principle, the system abstains from generating an ungrounded or speculative answer."*

---

## ⚠️ Limitations and Next Steps

### Current Limitations
1. **Complex Multi-Page Nested Tables**: Tables spanning across 3+ pages with merged multi-level row headers occasionally lose intermediate header associations during parsing.
2. **Chart / Vector Graphics Extraction**: Non-tabular infographic charts (e.g., pie charts embedded as raster images without embedded text) rely on OCR/vision rather than native vector layout blocks.
3. **Storage Concurrency**: SQLite handles high read concurrency seamlessly via WAL mode, but concurrent writes during multi-user bulk PDF uploads should transition to PostgreSQL for enterprise production.

### What We Would Build Next
1. **Cross-Encoder Re-Ranking Model**: Integrate a dedicated cross-encoder (e.g., `bge-reranker-large`) after RRF to further refine candidate ranking for multi-hop complex queries.
2. **Interactive Bounding Box Highlighting**: In addition to deep-linking the PDF viewer to the cited page (`#page=X`), render visual bounding box overlays highlighting the exact sentence or table row directly inside the PDF canvas.
3. **Automated Continuous Reconciliation Daemon**: An asynchronous worker that continuously compares newly ingested PDF facts against historical facts in the background, surfacing notifications when new quarterly reports contradict previous guidance.

---

## 📌 Additional Notes

- **Zero Hardcoding Guarantee**: The pipeline uses general semantic fact extraction. It will ingest any arbitrary PDF (legal, financial, medical, or government filings) without pre-defined schemas, hardcoded company names, or fixed column mappings.
- **Auditable Provenance**: Every fact row in SQLite retains its raw verbatim quote, bounding box coordinates, document SHA-256 hash, and 140-DPI visual page render path, providing a 100% auditable paper trail.
- **Repository**: [https://github.com/JKSANJAY27/Fact-Knowledge-Layer.git](https://github.com/JKSANJAY27/Fact-Knowledge-Layer.git)
