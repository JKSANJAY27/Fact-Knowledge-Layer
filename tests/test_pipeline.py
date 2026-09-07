import pytest
from fact_layer.normalization import (
    normalize_unit_and_value,
    normalize_temporal_expression,
    parse_numeric_value
)
from fact_layer.models import AtomicFact, TemporalContext, SourcePointer, QueryRequest
from fact_layer.reconciliation import reconciler, canonical_entity_name, metric_similarity
from fact_layer.retrieval import retriever
from fact_layer.storage import storage


def test_numeric_parsing():
    assert parse_numeric_value("8,142.17") == 8142.17
    assert parse_numeric_value("(452)") == -452.0
    assert parse_numeric_value("-1,008") == -1008.0
    assert parse_numeric_value("₹578 Cr") == 578.0


def test_unit_normalization():
    # Crore conversion
    v_cr, u_cr = normalize_unit_and_value("8,142.17", "INR Crore")
    assert v_cr == 81421700000.0
    assert u_cr == "INR"

    # Lakh conversion
    v_lakh, u_lakh = normalize_unit_and_value("50", "Lakhs")
    assert v_lakh == 5000000.0

    # Percentage
    v_pct, u_pct = normalize_unit_and_value("8.2%", "%")
    assert abs(v_pct - 0.082) < 1e-6
    assert u_pct == "%"

    # Basis points (120 bps = 0.012)
    v_bps, u_bps = normalize_unit_and_value("120", "BPS")
    assert abs(v_bps - 0.012) < 1e-6
    assert u_bps == "BPS"


def test_temporal_normalization():
    # Equivalence: FY2024, FY 2023-24, and Year ended March 31 2024
    p1 = normalize_temporal_expression("FY2024")
    p2 = normalize_temporal_expression("FY 2023-24")
    p3 = normalize_temporal_expression("Year ended March 31 2024")

    assert p1.canonical_period == "FY2024"
    assert p2.canonical_period == "FY2024"
    assert p3.canonical_period == "FY2024"
    assert p1.start_date == "2023-04-01"
    assert p1.end_date == "2024-03-31"

    # Quarter
    q4 = normalize_temporal_expression("Q4 FY24")
    assert q4.canonical_period == "Q4_FY2024"
    assert q4.start_date == "2024-01-01"
    assert q4.end_date == "2024-03-31"


def test_reconciliation_deterministic_cascade():
    def create_dummy_fact(fid, metric, raw_val, norm_val, period, scope, doc="Doc1", unit="INR Crore"):
        return AtomicFact(
            fact_id=fid,
            document_id="d1",
            page_number=1,
            block_id="b1",
            entity="Delhivery Limited",
            metric=metric,
            raw_value=raw_val,
            unit=unit,
            normalized_value=norm_val,
            canonical_unit="INR",
            temporal_context=TemporalContext(
                period_type="fiscal_year",
                start_date="2023-04-01",
                end_date="2024-03-31",
                canonical_period=period
            ),
            scope=scope,
            source_pointer=SourcePointer(
                document_id="d1",
                document_name=doc,
                page_number=1,
                block_id="b1",
                quoted_text="quote"
            ),
            confidence_score=1.0
        )

    f_base = create_dummy_fact("f1", "Revenue", "8,142.17", 81421700000.0, "FY2024", "consolidated", "Annual Report")
    f_agree = create_dummy_fact("f2", "Total Revenue", "8,142", 81420000000.0, "FY2024", "consolidated", "Presentation")
    f_disagree = create_dummy_fact("f3", "Revenue", "9,200", 92000000000.0, "FY2024", "consolidated", "Press Release")
    f_scope_diff = create_dummy_fact("f4", "Revenue", "7,800", 78000000000.0, "FY2024", "standalone", "Standalone")
    f_period_diff = create_dummy_fact("f5", "Revenue", "7,224", 72240000000.0, "FY2023", "consolidated", "FY23 Doc")

    # 1. Corroborated check
    r_corr = reconciler._evaluate_pair(f_base, f_agree, 0.95)
    assert r_corr.relation_type == "CORROBORATED"
    assert "Corroborated across disclosures" in r_corr.explanation

    # 2. Contradicted check
    r_contra = reconciler._evaluate_pair(f_base, f_disagree, 0.95)
    assert r_contra.relation_type == "CONTRADICTED"
    assert "Likely contradiction" in r_contra.explanation

    # 3. Contextual difference (Scope)
    r_scope = reconciler._evaluate_pair(f_base, f_scope_diff, 0.95)
    assert r_scope.relation_type == "CONTEXTUAL_DIFFERENCE"
    assert r_scope.difference_field == "scope"

    # 4. Contextual difference (Period)
    r_period = reconciler._evaluate_pair(f_base, f_period_diff, 0.95)
    assert r_period.relation_type == "CONTEXTUAL_DIFFERENCE"
    assert r_period.difference_field == "period"


def test_retriever_abstention():
    # Answering an ungrounded query where no facts exist
    res = retriever.answer_query(QueryRequest(query="What was Apple iPhone sales in Antarctica in 1850?"))
    assert "abstain" in res.answer.lower() or "no verified atomic facts" in res.answer.lower()
    assert len(res.grounded_facts) == 0


def test_storage_stats():
    stats = storage.get_stats()
    assert stats["total_documents"] >= 1
    assert stats["total_pages"] >= 1
    assert stats["total_facts"] >= 1
    assert stats["total_failures"] >= 1
