"""
Re-run LLM extraction on already-ingested blocks (without re-parsing PDFs).
This enriches the knowledge base with more atomic facts using the Gemini API.
"""
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from fact_layer.storage import storage
from fact_layer.extraction import extractor
from fact_layer.reconciliation import reconciler


def run_llm_extraction():
    print("=== LLM Extraction Pass (using pre-ingested blocks) ===", flush=True)

    documents = storage.list_documents()
    if not documents:
        print("No documents ingested yet. Run ingest_starter_data.py first.", flush=True)
        return

    total_facts = 0
    total_fails = 0

    for doc in documents:
        print(f"\nProcessing: {doc.filename} ({doc.document_id})", flush=True)
        pages = storage.get_pages_for_doc(doc.document_id)
        doc_facts = 0
        doc_fails = 0

        for p in pages:
            blocks = storage.get_blocks_for_page(doc.document_id, p.page_number)
            if not blocks:
                continue

            facts, fails = extractor.extract_facts_from_blocks(
                blocks,
                doc.document_id,
                doc.filename,
                p.page_number,
                use_llm=True  # LLM-powered extraction
            )
            doc_facts += len(facts)
            doc_fails += len(fails)

            if facts or fails:
                print(f"  Page {p.page_number}: {len(facts)} facts, {len(fails)} abstentions", flush=True)

            # Small delay to avoid rate limiting
            time.sleep(0.3)

        print(f"  [OK] {doc.filename}: {doc_facts} facts, {doc_fails} abstentions", flush=True)
        total_facts += doc_facts
        total_fails += doc_fails

    print(f"\n=== Running Cross-Document Reconciliation ===", flush=True)
    rels = reconciler.reconcile_facts()
    print(f"Total Relationships: {len(rels)}", flush=True)

    stats = storage.get_stats()
    print("\n=== Updated Knowledge Layer Stats ===", flush=True)
    print(f"  Documents      : {stats['total_documents']}")
    print(f"  Pages          : {stats['total_pages']}")
    print(f"  Atomic Facts   : {stats['total_facts']}")
    print(f"  Relationships  : {stats['relationships']}")
    print(f"  Abstentions    : {stats['failures']}")


if __name__ == "__main__":
    run_llm_extraction()
