"""
Ingest all starter dataset PDFs fully (no page limit).
Schema-free: the pipeline discovers facts from document content — no hardcoded rules,
filenames, or document-specific logic. Works for any PDF placed in starter-datasets/.
"""
from pathlib import Path
import shutil
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from fact_layer.config import STARTER_DATASETS_DIR, PDFS_DIR
from fact_layer.ingestion import ingestor
from fact_layer.extraction import extractor
from fact_layer.reconciliation import reconciler
from fact_layer.storage import storage


def copy_pdf_for_serving(src: Path):
    """Copy PDF to PDFS_DIR so it can be served via /static/pdfs/."""
    dest = PDFS_DIR / src.name
    if not dest.exists():
        shutil.copy2(src, dest)
        print(f"  Copied {src.name} -> {dest}", flush=True)


def run_starter_ingestion(max_pages: int = None, use_llm: bool = False):
    print(
        f"=== Starting Full Starter Dataset Ingestion "
        f"(max_pages={'ALL' if max_pages is None else max_pages}, use_llm={use_llm}) ===",
        flush=True
    )

    datasets = [
        STARTER_DATASETS_DIR / "delhivery",
        STARTER_DATASETS_DIR / "india-macroeconomy"
    ]

    for d in datasets:
        if not d.exists():
            print(f"Directory not found: {d}", flush=True)
            continue

        for pdf in sorted(d.glob("*.pdf")):
            print(f"\nProcessing {pdf.name} ...", flush=True)
            try:
                # Make the PDF available for the citation viewer
                copy_pdf_for_serving(pdf)

                # Ingest: parse every page into blocks (text, tables, layout)
                doc_rec, pages, blocks = ingestor.ingest_pdf(
                    pdf, max_pages=max_pages, force_reprocess=False
                )
                print(
                    f"  Ingested: {doc_rec.filename} -> {len(pages)} pages, {len(blocks)} blocks",
                    flush=True
                )

                doc_facts = 0
                doc_fails = 0
                for p in pages:
                    p_blocks = storage.get_blocks_for_page(doc_rec.document_id, p.page_number)
                    facts, fails = extractor.extract_facts_from_blocks(
                        p_blocks,
                        doc_rec.document_id,
                        doc_rec.filename,
                        p.page_number,
                        use_llm=use_llm
                    )
                    doc_facts += len(facts)
                    doc_fails += len(fails)
                    print(
                        f"    Page {p.page_number}: {len(facts)} facts, {len(fails)} abstentions",
                        flush=True
                    )

                print(
                    f"  [OK] Total: {doc_facts} facts, {doc_fails} abstentions/failures",
                    flush=True
                )
            except Exception as e:
                print(f"  [ERR] Error processing {pdf.name}: {e}", flush=True)
                import traceback; traceback.print_exc()

    print("\n=== Running Cross-Document Reconciliation ===", flush=True)
    rels = reconciler.reconcile_facts()
    print(f"Total Relationships Established: {len(rels)}", flush=True)

    stats = storage.get_stats()
    print("\n=== Final Knowledge Layer Stats ===", flush=True)
    print(f"  Documents  : {stats['total_documents']}")
    print(f"  Pages      : {stats['total_pages']}")
    print(f"  Blocks     : {stats.get('total_blocks', '?')}")
    print(f"  Atomic Facts: {stats['total_facts']}")
    print(f"  Relationships: {stats['relationships']}")
    print(f"  Failures/Abstentions: {stats['failures']}")


if __name__ == "__main__":
    # Full ingestion — no page limit, rule-based extraction (fast)
    # Set use_llm=True to use LLM-enhanced extraction (slower but richer)
    run_starter_ingestion(max_pages=None, use_llm=False)
