"""Quick test of the 4 case example queries against the live knowledge base."""
import sys, json
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from fact_layer.models import QueryRequest
from fact_layer.retrieval import retriever
from fact_layer.storage import storage

QUERIES = [
    ("CORROBORATED", "What is India's GDP growth rate and is it consistent across documents?"),
    ("CONTRADICTED", "What does each document say about Delhivery's revenue or profitability — do they agree?"),
    ("CONTEXTUAL", "Are there conflicting inflation figures across reports, and can these be reconciled by time period or scope?"),
    ("ABSTENTION", "What is the exact unit and measurement basis for Delhivery's shipment volume?"),
]

print(f"=== Knowledge Base: {storage.get_stats()} ===\n")

for case, query in QUERIES:
    print(f"\n{'='*60}")
    print(f"[{case}] {query}")
    print("="*60)
    req = QueryRequest(query=query, top_k=8)
    resp = retriever.answer_query(req)
    print(f"ANSWER:\n{resp.answer}")
    print(f"\nCITATIONS ({len(resp.grounded_facts)}):")
    for c in resp.grounded_facts[:4]:
        print(f"  - {c.entity} | {c.metric} = {c.raw_value} | {c.document_name} p{c.page_number}")
    print(f"\nRELATIONSHIPS ({len(resp.related_reconciliations)}):")
    for r in resp.related_reconciliations[:3]:
        print(f"  - {r['relation_type']}: {r.get('explanation','')[:80]}")
    print(f"\nCONFIDENCE NOTE: {resp.confidence_note}")
