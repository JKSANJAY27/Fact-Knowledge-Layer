from __future__ import annotations
import os
import shutil
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from fact_layer.config import (
    PAGES_DIR,
    UPLOADS_DIR,
    PDFS_DIR,
    DATA_DIR,
    STARTER_DATASETS_DIR
)
from fact_layer.models import (
    DocumentRecord,
    PageRecord,
    BlockRecord,
    AtomicFact,
    FactRelationship,
    ExtractionFailure,
    FactFilter,
    QueryRequest,
    QueryResponse
)
from fact_layer.storage import storage
from fact_layer.ingestion import ingestor
from fact_layer.extraction import extractor
from fact_layer.reconciliation import reconciler
from fact_layer.retrieval import retriever

app = FastAPI(
    title="Fact Knowledge Layer API",
    description="Atomic fact extraction, normalization, deterministic reconciliation, and grounded retrieval for financial and institutional documents.",
    version="1.0.0"
)

# Enable CORS for Next.js frontend (localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve rendered page images statically at /static/pages/
if not PAGES_DIR.exists():
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/pages", StaticFiles(directory=str(PAGES_DIR)), name="pages")

# Serve raw PDFs at /static/pdfs/ for the citation viewer
if not PDFS_DIR.exists():
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/pdfs", StaticFiles(directory=str(PDFS_DIR)), name="pdfs")


# --------------------------------------------------------------------------- #
# In-memory job tracker for upload progress
# --------------------------------------------------------------------------- #
_upload_jobs: Dict[str, Dict[str, Any]] = {}


def _copy_pdf_for_serving(src: Path) -> str:
    """Copies a PDF to PDFS_DIR so it can be served via /static/pdfs/. Returns filename."""
    dest = PDFS_DIR / src.name
    if not dest.exists():
        shutil.copy2(src, dest)
    return src.name


def process_document_pipeline(
    file_path: Path,
    max_pages: Optional[int] = None,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    """Runs the full ingestion → extraction → normalization → reconciliation pipeline on a file.
    Entirely schema-free: facts are discovered by the LLM from whatever content appears in the PDF.
    """
    if job_id:
        _upload_jobs[job_id] = {"status": "ingesting", "progress": 0}

    # 1. Ingest: parse PDF into pages + blocks (text, tables, layout)
    doc_rec, pages, blocks = ingestor.ingest_pdf(file_path, max_pages=max_pages, force_reprocess=False)

    if job_id:
        _upload_jobs[job_id]["status"] = "extracting"

    # 2. Extract atomic facts page by page (LLM-agnostic schema — the model decides what a fact is)
    all_facts: List[AtomicFact] = []
    all_fails: List[ExtractionFailure] = []

    for i, p in enumerate(pages):
        p_blocks = storage.get_blocks_for_page(doc_rec.document_id, p.page_number)
        facts, fails = extractor.extract_facts_from_blocks(
            p_blocks,
            doc_rec.document_id,
            doc_rec.filename,
            p.page_number
        )
        all_facts.extend(facts)
        all_fails.extend(fails)
        if job_id:
            _upload_jobs[job_id]["progress"] = int((i + 1) / len(pages) * 80)

    if job_id:
        _upload_jobs[job_id]["status"] = "reconciling"

    # 3. Run cross-document reconciliation to discover corroborations, contradictions, etc.
    relationships = reconciler.reconcile_facts()

    if job_id:
        _upload_jobs[job_id] = {
            "status": "done",
            "progress": 100,
            "result": {
                "document_id": doc_rec.document_id,
                "filename": doc_rec.filename,
                "pages_processed": len(pages),
                "facts_extracted": len(all_facts),
                "failures_logged": len(all_fails),
                "total_relationships": len(relationships)
            }
        }

    return {
        "document_id": doc_rec.document_id,
        "filename": doc_rec.filename,
        "pages_processed": len(pages),
        "facts_extracted": len(all_facts),
        "failures_logged": len(all_fails),
        "total_relationships": len(relationships)
    }


# =========================================================================== #
# Health & Stats
# =========================================================================== #

@app.get("/api/health")
def health():
    return {"status": "healthy", "service": "fact-knowledge-layer"}


@app.get("/api/stats")
def get_stats():
    return storage.get_stats()


# =========================================================================== #
# Documents
# =========================================================================== #

@app.get("/api/documents", response_model=List[DocumentRecord])
def list_documents():
    return storage.list_documents()


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str):
    doc = storage.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.get("/api/documents/{doc_id}/pages", response_model=List[PageRecord])
def list_document_pages(doc_id: str):
    return storage.get_pages_for_doc(doc_id)


@app.get("/api/documents/{doc_id}/pages/{page_num}")
def get_page_details(doc_id: str, page_num: int):
    page = storage.get_page(doc_id, page_num)
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")

    blocks = storage.get_blocks_for_page(doc_id, page_num)
    facts = storage.list_facts(FactFilter(document_id=doc_id, limit=200))
    page_facts = [f for f in facts if f.page_number == page_num]

    # Build the page image URL — pages are stored as {doc_id}_p{page_num}.png
    image_url = f"/static/pages/{doc_id}_p{page_num}.png"

    return {
        "page": page,
        "blocks": blocks,
        "facts": page_facts,
        "image_url": image_url
    }


# =========================================================================== #
# Facts, Relationships, Failures
# =========================================================================== #

@app.get("/api/facts", response_model=List[AtomicFact])
def list_facts(
    entity: Optional[str] = None,
    metric: Optional[str] = None,
    document_id: Optional[str] = None,
    canonical_period: Optional[str] = None,
    scope: Optional[str] = None,
    min_confidence: Optional[float] = None,
    search_query: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    filter_params = FactFilter(
        entity=entity,
        metric=metric,
        document_id=document_id,
        canonical_period=canonical_period,
        scope=scope,
        min_confidence=min_confidence,
        search_query=search_query,
        limit=limit,
        offset=offset
    )
    return storage.list_facts(filter_params)


@app.get("/api/facts/{fact_id}")
def get_fact_details(fact_id: str):
    fact = storage.get_fact(fact_id)
    if not fact:
        raise HTTPException(status_code=404, detail="Fact not found")

    block = storage.get_block(fact.block_id)
    all_rels = storage.list_relationships(limit=200)
    related = [r for r in all_rels if r.fact_a_id == fact_id or r.fact_b_id == fact_id]

    return {
        "fact": fact,
        "block": block,
        "relationships": related,
        "image_url": f"/static/pages/{fact.document_id}_p{fact.page_number}.png"
    }


@app.get("/api/relationships", response_model=List[FactRelationship])
def list_relationships(
    relation_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    return storage.list_relationships(relation_type=relation_type, limit=limit, offset=offset)


@app.get("/api/failures", response_model=List[ExtractionFailure])
def list_failures(
    failure_type: Optional[str] = None,
    limit: int = 100
):
    return storage.list_failures(failure_type=failure_type, limit=limit)


# =========================================================================== #
# Query / RAG
# =========================================================================== #

@app.post("/api/query", response_model=QueryResponse)
def query_facts(req: QueryRequest):
    return retriever.answer_query(req)


# =========================================================================== #
# Upload — generic PDF ingestion (schema-free, works for any document)
# =========================================================================== #

@app.post("/api/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    max_pages: Optional[int] = Form(None)
):
    """Uploads any PDF file and runs the full ingestion pipeline asynchronously.
    Schema-free: the fact extractor discovers structure from document content, no hardcoding."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    import uuid
    job_id = str(uuid.uuid4())[:8]

    dest_path = UPLOADS_DIR / file.filename
    content = await file.read()
    with open(dest_path, "wb") as f:
        f.write(content)

    # Also copy to PDFS_DIR so it's immediately serveable via /static/pdfs/
    _copy_pdf_for_serving(dest_path)

    _upload_jobs[job_id] = {"status": "queued", "progress": 0}

    # Run pipeline in background thread so upload returns immediately
    def _run():
        process_document_pipeline(dest_path, max_pages=max_pages, job_id=job_id)

    background_tasks.add_task(_run)

    return {"job_id": job_id, "filename": file.filename, "status": "queued"}


@app.get("/api/documents/upload/status/{job_id}")
def upload_status(job_id: str):
    """Poll upload job progress."""
    job = _upload_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# =========================================================================== #
# Starter-dataset processing (generic — processes whatever PDFs are in the dirs)
# =========================================================================== #

class ProcessStarterRequest(BaseModel):
    dataset: str = "all"        # "delhivery", "india-macroeconomy", "all"
    max_pages_per_doc: Optional[int] = None   # None = all pages


@app.post("/api/pipeline/process-starter")
def process_starter_datasets(req: ProcessStarterRequest):
    """Processes all PDFs in starter-datasets/ directories.
    Fully generic: no hardcoded filenames, schemas, or document-specific rules."""
    results = []

    target_dirs = []
    if req.dataset in ["delhivery", "all"]:
        target_dirs.append(STARTER_DATASETS_DIR / "delhivery")
    if req.dataset in ["india-macroeconomy", "all"]:
        target_dirs.append(STARTER_DATASETS_DIR / "india-macroeconomy")

    for d in target_dirs:
        if not d.exists():
            continue
        for pdf_file in sorted(d.glob("*.pdf")):
            try:
                # Copy to PDFS_DIR for serving
                _copy_pdf_for_serving(pdf_file)
                res = process_document_pipeline(pdf_file, max_pages=req.max_pages_per_doc)
                results.append(res)
            except Exception as e:
                results.append({
                    "filename": pdf_file.name,
                    "status": "error",
                    "error": str(e)
                })

    # Final cross-document reconciliation pass
    reconciler.reconcile_facts()

    return {
        "processed_documents": results,
        "current_stats": storage.get_stats()
    }
