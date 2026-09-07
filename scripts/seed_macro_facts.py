"""
Seed high-quality macroeconomic facts extracted from India Economic Survey, RBI, and IMF documents,
as well as complete Delhivery company facts.
This enriches the knowledge base so that queries across all 6 documents return accurate, verified facts.
"""
import sys
import uuid
import json
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from fact_layer.storage import storage
from fact_layer.models import (
    AtomicFact,
    TemporalContext,
    SourcePointer
)
from fact_layer.reconciliation import reconciler


def seed_macro_and_delhivery_facts():
    print("Seeding macroeconomic and institutional facts into database...", flush=True)

    # Fetch document IDs
    doc_map = {d.filename: d.document_id for d in storage.list_documents()}

    econ_doc_id = doc_map.get("01-india-economic-survey-2024-25-excerpt.pdf", "doc_econ")
    rbi_doc_id = doc_map.get("02-rbi-annual-report-2024-25-excerpt.pdf", "doc_rbi")
    imf_doc_id = doc_map.get("03-imf-india-2025-article-iv-excerpt.pdf", "doc_imf")
    delhivery_q4_id = doc_map.get("03-delhivery-q4-fy24-earnings-presentation.pdf", "doc_delh_q4")
    delhivery_ar_id = doc_map.get("02-delhivery-annual-report-fy24-excerpt.pdf", "doc_delh_ar")
    delhivery_pro_id = doc_map.get("01-delhivery-prospectus-2022-excerpt.pdf", "doc_delh_pro")

    new_facts = [
        # --- INDIA MACROECONOMY (Economic Survey) ---
        AtomicFact(
            fact_id=f"fact_macro_{uuid.uuid4().hex[:8]}",
            document_id=econ_doc_id,
            page_number=14,
            block_id=f"blk_{econ_doc_id}_p14_gdp",
            entity="Government of India",
            metric="Real GDP Growth",
            raw_value="6.4",
            unit="%",
            normalized_value=0.064,
            canonical_unit="ratio",
            temporal_context=TemporalContext(
                period_type="fiscal_year",
                start_date="2024-04-01",
                end_date="2025-03-31",
                canonical_period="FY2025",
                raw_period_text="FY25"
            ),
            scope="institutional",
            source_pointer=SourcePointer(
                document_id=econ_doc_id,
                document_name="01-india-economic-survey-2024-25-excerpt.pdf",
                page_number=14,
                block_id=f"blk_{econ_doc_id}_p14_gdp",
                quoted_text="the real gross domestic product (GDP) growth for FY25 is estimated to be 6.4 per cent"
            ),
            confidence_score=0.96,
            confidence_breakdown={"clarity": 0.98, "evidence_match": 0.98, "normalization": 0.95}
        ),
        AtomicFact(
            fact_id=f"fact_macro_{uuid.uuid4().hex[:8]}",
            document_id=econ_doc_id,
            page_number=14,
            block_id=f"blk_{econ_doc_id}_p14_pfce",
            entity="Government of India",
            metric="Private Final Consumption Expenditure Growth",
            raw_value="7.3",
            unit="%",
            normalized_value=0.073,
            canonical_unit="ratio",
            temporal_context=TemporalContext(
                period_type="fiscal_year",
                start_date="2024-04-01",
                end_date="2025-03-31",
                canonical_period="FY2025",
                raw_period_text="FY25"
            ),
            scope="institutional",
            source_pointer=SourcePointer(
                document_id=econ_doc_id,
                document_name="01-india-economic-survey-2024-25-excerpt.pdf",
                page_number=14,
                block_id=f"blk_{econ_doc_id}_p14_pfce",
                quoted_text="private final consumption expenditure at constant prices is estimated to grow by 7.3 per cent"
            ),
            confidence_score=0.95,
            confidence_breakdown={"clarity": 0.95, "evidence_match": 0.95, "normalization": 0.95}
        ),

        # --- INDIA MACROECONOMY (IMF Article IV) ---
        AtomicFact(
            fact_id=f"fact_macro_{uuid.uuid4().hex[:8]}",
            document_id=imf_doc_id,
            page_number=5,
            block_id=f"blk_{imf_doc_id}_p5_gdp",
            entity="Government of India",
            metric="Real GDP Growth",
            raw_value="6.5",
            unit="%",
            normalized_value=0.065,
            canonical_unit="ratio",
            temporal_context=TemporalContext(
                period_type="fiscal_year",
                start_date="2024-04-01",
                end_date="2025-03-31",
                canonical_period="FY2025",
                raw_period_text="2024/25"
            ),
            scope="institutional",
            source_pointer=SourcePointer(
                document_id=imf_doc_id,
                document_name="03-imf-india-2025-article-iv-excerpt.pdf",
                page_number=5,
                block_id=f"blk_{imf_doc_id}_p5_gdp",
                quoted_text="Real GDP (at market prices) 2024/25: 6.5"
            ),
            confidence_score=0.96,
            confidence_breakdown={"clarity": 0.98, "evidence_match": 0.96, "normalization": 0.95}
        ),
        AtomicFact(
            fact_id=f"fact_macro_{uuid.uuid4().hex[:8]}",
            document_id=imf_doc_id,
            page_number=5,
            block_id=f"blk_{imf_doc_id}_p5_inflation",
            entity="Government of India",
            metric="Consumer Price Inflation",
            raw_value="4.6",
            unit="%",
            normalized_value=0.046,
            canonical_unit="ratio",
            temporal_context=TemporalContext(
                period_type="fiscal_year",
                start_date="2024-04-01",
                end_date="2025-03-31",
                canonical_period="FY2025",
                raw_period_text="2024/25"
            ),
            scope="institutional",
            source_pointer=SourcePointer(
                document_id=imf_doc_id,
                document_name="03-imf-india-2025-article-iv-excerpt.pdf",
                page_number=5,
                block_id=f"blk_{imf_doc_id}_p5_inflation",
                quoted_text="Consumer prices - Combined 2024/25: 4.6"
            ),
            confidence_score=0.95,
            confidence_breakdown={"clarity": 0.95, "evidence_match": 0.95, "normalization": 0.95}
        ),

        # --- INDIA MACROECONOMY (RBI Annual Report) ---
        AtomicFact(
            fact_id=f"fact_macro_{uuid.uuid4().hex[:8]}",
            document_id=rbi_doc_id,
            page_number=17,
            block_id=f"blk_{rbi_doc_id}_p17_gdp",
            entity="Government of India",
            metric="Real GDP Growth",
            raw_value="6.5",
            unit="%",
            normalized_value=0.065,
            canonical_unit="ratio",
            temporal_context=TemporalContext(
                period_type="fiscal_year",
                start_date="2025-04-01",
                end_date="2026-03-31",
                canonical_period="FY2026",
                raw_period_text="2025-26"
            ),
            scope="institutional",
            source_pointer=SourcePointer(
                document_id=rbi_doc_id,
                document_name="02-rbi-annual-report-2024-25-excerpt.pdf",
                page_number=17,
                block_id=f"blk_{rbi_doc_id}_p17_gdp",
                quoted_text="real GDP growth for 2025-26 is projected at 6.5 per cent, with risks evenly balanced"
            ),
            confidence_score=0.95,
            confidence_breakdown={"clarity": 0.95, "evidence_match": 0.95, "normalization": 0.95}
        ),

        # --- DELHIVERY PROSPECTUS & ANNUAL REPORT ---
        AtomicFact(
            fact_id=f"fact_delh_{uuid.uuid4().hex[:8]}",
            document_id=delhivery_ar_id,
            page_number=2,
            block_id=f"blk_{delhivery_ar_id}_p2_pin",
            entity="Delhivery Limited",
            metric="PIN Code Reach",
            raw_value="18,792",
            unit="Pin Codes",
            normalized_value=18792.0,
            canonical_unit="count",
            temporal_context=TemporalContext(
                period_type="fiscal_year",
                start_date="2023-04-01",
                end_date="2024-03-31",
                canonical_period="FY2024",
                raw_period_text="FY24"
            ),
            scope="consolidated",
            source_pointer=SourcePointer(
                document_id=delhivery_ar_id,
                document_name="02-delhivery-annual-report-fy24-excerpt.pdf",
                page_number=2,
                block_id=f"blk_{delhivery_ar_id}_p2_pin",
                quoted_text="Covering 18,792 active pin codes across India representing over 90% of Indian population"
            ),
            confidence_score=0.98,
            confidence_breakdown={"clarity": 0.98, "evidence_match": 0.98, "normalization": 0.98}
        ),
        AtomicFact(
            fact_id=f"fact_delh_{uuid.uuid4().hex[:8]}",
            document_id=delhivery_ar_id,
            page_number=2,
            block_id=f"blk_{delhivery_ar_id}_p2_revenue",
            entity="Delhivery Limited",
            metric="Revenue from Contracts with Customers",
            raw_value="8,142",
            unit="INR Crore",
            normalized_value=81420000000.0,
            canonical_unit="INR",
            temporal_context=TemporalContext(
                period_type="fiscal_year",
                start_date="2023-04-01",
                end_date="2024-03-31",
                canonical_period="FY2024",
                raw_period_text="FY2023-24"
            ),
            scope="consolidated",
            source_pointer=SourcePointer(
                document_id=delhivery_ar_id,
                document_name="02-delhivery-annual-report-fy24-excerpt.pdf",
                page_number=2,
                block_id=f"blk_{delhivery_ar_id}_p2_revenue",
                quoted_text="Revenue from contracts with customers reached ₹8,142 Crores in FY24"
            ),
            confidence_score=0.98,
            confidence_breakdown={"clarity": 0.98, "evidence_match": 0.98, "normalization": 0.98}
        ),
        AtomicFact(
            fact_id=f"fact_delh_{uuid.uuid4().hex[:8]}",
            document_id=delhivery_ar_id,
            page_number=3,
            block_id=f"blk_{delhivery_ar_id}_p3_shipments",
            entity="Delhivery Limited",
            metric="Express Parcel Shipments",
            raw_value="740",
            unit="Million Packages",
            normalized_value=740000000.0,
            canonical_unit="count",
            temporal_context=TemporalContext(
                period_type="fiscal_year",
                start_date="2023-04-01",
                end_date="2024-03-31",
                canonical_period="FY2024",
                raw_period_text="FY24"
            ),
            scope="consolidated",
            source_pointer=SourcePointer(
                document_id=delhivery_ar_id,
                document_name="02-delhivery-annual-report-fy24-excerpt.pdf",
                page_number=3,
                block_id=f"blk_{delhivery_ar_id}_p3_shipments",
                quoted_text="Handled over 740 million express parcel shipments in FY24"
            ),
            confidence_score=0.97,
            confidence_breakdown={"clarity": 0.97, "evidence_match": 0.97, "normalization": 0.97}
        )
    ]

    storage.add_facts(new_facts)
    print(f"Added {len(new_facts)} atomic facts.", flush=True)

    # Run reconciliation across new facts
    rels = reconciler.reconcile_facts()
    print(f"Total relationships reconciled: {len(rels)}", flush=True)

    # Sync to seed_data.sqlite3
    import shutil
    shutil.copy2(storage.db_path, project_root / "data" / "seed_data.sqlite3")
    print("Synced updated database to data/seed_data.sqlite3", flush=True)

    stats = storage.get_stats()
    print(f"Updated Stats: {stats}", flush=True)


if __name__ == "__main__":
    seed_macro_and_delhivery_facts()
