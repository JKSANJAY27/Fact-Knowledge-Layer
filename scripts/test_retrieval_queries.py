import sys
sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
from fact_layer.retrieval import retriever
from fact_layer.models import QueryRequest

queries = [
    "What was NovaCorp's consolidated revenue for FY2024?",
    "What is NovaCorp's total verified workforce or headcount?",
    "How does NovaCorp's standalone revenue compare to its consolidated revenue?",
    "What is NovaCorp's projected market expansion index?",
    "What was Tesla's revenue in 2024?",  # Unknown company -> must abstain!
]

for q in queries:
    print(f"\n==========================================")
    print(f"QUERY: {q}")
    res = retriever.answer_query(QueryRequest(query=q))
    print(f"ANSWER:\n{res.answer}")
    print(f"CITATIONS ({len(res.grounded_facts)}):")
    for f in res.grounded_facts[:3]:
        print(f"  - {f.entity} | {f.metric} = {f.raw_value} | {f.canonical_period} | {f.scope} (p.{f.page_number})")
    print(f"RELATIONSHIPS ({len(res.related_reconciliations)}):")
    for r in res.related_reconciliations[:2]:
        print(f"  - {r.get('relation_type')}: {r.get('difference_field')}")
    print(f"CONFIDENCE NOTE: {res.confidence_note}")
