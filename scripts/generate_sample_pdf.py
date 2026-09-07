"""
Generates a clean, realistic 2-page financial & operational PDF for testing.
Contains facts suitable for demonstrating:
1. Corroboration: Headcount / Employee count (8,500) reported identically across sections.
2. Contradiction / Period variance: FY2023 (INR 980 Cr) vs FY2024 (INR 1,250 Cr).
3. Contextual discrepancy: Standalone Revenue (INR 1,120 Cr) vs Consolidated Revenue (INR 1,250 Cr).
4. Abstention: Projected Market Expansion Index (7.8, unspecified baseline and unit).
"""
from pathlib import Path
import fitz  # PyMuPDF


def generate_sample_pdf(output_path: Path):
    doc = fitz.open()

    # --- PAGE 1 ---
    page1 = doc.new_page(width=595, height=842)  # A4 size

    # Header / Title banner
    page1.draw_rect(fitz.Rect(40, 40, 555, 90), color=(0.1, 0.2, 0.45), fill=(0.93, 0.95, 0.99))
    page1.insert_text((55, 62), "NovaCorp Technologies Limited", fontsize=15, fontname="helv", color=(0.1, 0.2, 0.45))
    page1.insert_text((55, 78), "Annual Financial & Operational Review — FY2024", fontsize=10, fontname="helv", color=(0.3, 0.4, 0.5))

    # Section 1: Executive Overview
    y = 115
    page1.insert_text((40, y), "1. Executive Summary & Financial Highlights", fontsize=13, fontname="helv", color=(0.1, 0.15, 0.3))
    y += 18
    summary_text = (
        "NovaCorp Technologies Limited delivered robust growth in Fiscal Year 2024, driven by accelerated "
        "digital logistics and cloud enterprise solutions. Consolidated revenue for FY2024 reached INR 1,250 Crores, "
        "representing a 27.5% year-on-year expansion compared to INR 980 Crores in FY2023. Operating profitability "
        "improved substantially, with Adjusted EBITDA reaching INR 225 Crores, representing an operating margin of 18.0%."
    )
    page1.insert_textbox(fitz.Rect(40, y, 555, y + 60), summary_text, fontsize=9.5, fontname="helv", color=(0.2, 0.2, 0.2), lineheight=1.3)
    y += 75

    # Financial Performance Table
    page1.insert_text((40, y), "Table 1.1: Consolidated Financial Performance Metrics", fontsize=10.5, fontname="helv", color=(0.15, 0.2, 0.3))
    y += 12

    # Draw table outline & rows
    table_rect = fitz.Rect(40, y, 555, y + 105)
    page1.draw_rect(table_rect, color=(0.7, 0.75, 0.8), width=0.8)
    
    # Table Header
    page1.draw_rect(fitz.Rect(40, y, 555, y + 22), color=(0.7, 0.75, 0.8), fill=(0.88, 0.91, 0.96))
    page1.insert_text((50, y + 15), "Metric", fontsize=9, fontname="helv", color=(0.1, 0.15, 0.3))
    page1.insert_text((230, y + 15), "FY2024 (Consolidated)", fontsize=9, fontname="helv", color=(0.1, 0.15, 0.3))
    page1.insert_text((370, y + 15), "FY2023 (Consolidated)", fontsize=9, fontname="helv", color=(0.1, 0.15, 0.3))
    page1.insert_text((485, y + 15), "YoY Change", fontsize=9, fontname="helv", color=(0.1, 0.15, 0.3))

    rows = [
        ("Revenue from Operations", "INR 1,250 Cr", "INR 980 Cr", "+27.5%"),
        ("Adjusted EBITDA", "INR 225 Cr", "INR 155 Cr", "+45.1%"),
        ("EBITDA Margin", "18.0%", "15.8%", "+220 bps"),
        ("Active Full-Time Employees", "8,500", "7,100", "+19.7%"),
    ]

    for i, (m, f24, f23, chg) in enumerate(rows):
        row_y = y + 22 + (i * 20)
        page1.draw_line((40, row_y), (555, row_y), color=(0.85, 0.88, 0.92))
        page1.insert_text((50, row_y + 14), m, fontsize=8.5, fontname="helv", color=(0.2, 0.2, 0.2))
        page1.insert_text((240, row_y + 14), f24, fontsize=8.5, fontname="helv", color=(0.1, 0.1, 0.1))
        page1.insert_text((380, row_y + 14), f23, fontsize=8.5, fontname="helv", color=(0.1, 0.1, 0.1))
        page1.insert_text((495, row_y + 14), chg, fontsize=8.5, fontname="helv", color=(0.1, 0.5, 0.2))

    y += 125

    # Section 2: Scope Context — Standalone vs Consolidated Discrepancy
    page1.insert_text((40, y), "2. Legal Entity & Scope Reconciliation", fontsize=13, fontname="helv", color=(0.1, 0.15, 0.3))
    y += 18
    scope_text = (
        "NovaCorp Technologies Limited reports both consolidated accounts (incorporating international logistics "
        "subsidiaries across Singapore and Dubai) and standalone accounts for core domestic operations. "
        "Standalone revenue for FY2024 stood at INR 1,120 Crores, compared to the consolidated figure of INR 1,250 Crores. "
        "The difference of INR 130 Crores corresponds directly to foreign operating subsidiaries. Readers should evaluate "
        "these figures in light of entity perimeter disclosures."
    )
    page1.insert_textbox(fitz.Rect(40, y, 555, y + 65), scope_text, fontsize=9.5, fontname="helv", color=(0.2, 0.2, 0.2), lineheight=1.3)
    y += 85

    # Section 3: Abstention Candidate
    page1.insert_text((40, y), "3. Outlook & Market Perception Indicator", fontsize=13, fontname="helv", color=(0.1, 0.15, 0.3))
    y += 18
    ambig_text = (
        "According to an internal management assessment, the company's projected market expansion index is rated at 7.8. "
        "Note: The measurement methodology, baseline index unit, and comparative industry benchmarks were not defined "
        "at the time of publication and remain uncertified."
    )
    page1.insert_textbox(fitz.Rect(40, y, 555, y + 50), ambig_text, fontsize=9.5, fontname="helv", color=(0.2, 0.2, 0.2), lineheight=1.3)

    # Footer Page 1
    page1.draw_line((40, 800), (555, 800), color=(0.8, 0.8, 0.8))
    page1.insert_text((40, 815), "NovaCorp Technologies Limited | FY2024 Review", fontsize=8, fontname="helv", color=(0.5, 0.5, 0.5))
    page1.insert_text((515, 815), "Page 1 of 2", fontsize=8, fontname="helv", color=(0.5, 0.5, 0.5))

    # --- PAGE 2 ---
    page2 = doc.new_page(width=595, height=842)

    # Header Page 2
    page2.draw_rect(fitz.Rect(40, 40, 555, 75), color=(0.1, 0.2, 0.45), fill=(0.96, 0.97, 0.99))
    page2.insert_text((55, 60), "NovaCorp Technologies Limited — Workforce & Geographic Reach", fontsize=12, fontname="helv", color=(0.1, 0.2, 0.45))

    y = 95
    page2.insert_text((40, y), "4. Human Capital and Workforce Corroboration", fontsize=13, fontname="helv", color=(0.1, 0.15, 0.3))
    y += 18
    wf_text = (
        "Human capital remains central to NovaCorp's engineering execution. As of March 31, 2024, the company "
        "employed a verified workforce of 8,500 full-time employees worldwide. Engineering, software development, "
        "and product architecture accounted for 54% of the total headcount, with customer logistics operations "
        "representing 36% and administrative services representing the remaining 10%."
    )
    page2.insert_textbox(fitz.Rect(40, y, 555, y + 60), wf_text, fontsize=9.5, fontname="helv", color=(0.2, 0.2, 0.2), lineheight=1.3)
    y += 75

    # Geographic Coverage & Tier-2 Expansion
    page2.insert_text((40, y), "5. Network Infrastructure and PIN Code Coverage", fontsize=13, fontname="helv", color=(0.1, 0.15, 0.3))
    y += 18
    reach_text = (
        "The national fulfillment network spanned 18,400 active PIN codes nationwide by the close of FY2024. "
        "During the year under review, the dedicated Tier-2 and Tier-3 rural expansion initiative added coverage "
        "across 1,200 new PIN codes. Stakeholders should distinguish between the total nationwide reach of 18,400 PIN codes "
        "and the incremental rural expansion reach of 1,200 PIN codes."
    )
    page2.insert_textbox(fitz.Rect(40, y, 555, y + 65), reach_text, fontsize=9.5, fontname="helv", color=(0.2, 0.2, 0.2), lineheight=1.3)
    y += 85

    # Summary table of operational facts
    page2.insert_text((40, y), "Table 2.1: Key Operational Statistics", fontsize=10.5, fontname="helv", color=(0.15, 0.2, 0.3))
    y += 12

    op_table = fitz.Rect(40, y, 555, y + 85)
    page2.draw_rect(op_table, color=(0.7, 0.75, 0.8), width=0.8)
    page2.draw_rect(fitz.Rect(40, y, 555, y + 20), color=(0.7, 0.75, 0.8), fill=(0.88, 0.91, 0.96))
    page2.insert_text((50, y + 14), "Operational Dimension", fontsize=9, fontname="helv", color=(0.1, 0.15, 0.3))
    page2.insert_text((270, y + 14), "Reported Value", fontsize=9, fontname="helv", color=(0.1, 0.15, 0.3))
    page2.insert_text((410, y + 14), "Verification Scope", fontsize=9, fontname="helv", color=(0.1, 0.15, 0.3))

    op_rows = [
        ("Total Global Headcount", "8,500 employees", "Consolidated (Page 1 & 2)"),
        ("Nationwide Network Coverage", "18,400 PIN codes", "Total Postal Reach"),
        ("Rural Expansion Increment", "1,200 PIN codes", "New FY2024 Phase 2"),
    ]

    for i, (dim, val, scp) in enumerate(op_rows):
        row_y = y + 20 + (i * 20)
        page2.draw_line((40, row_y), (555, row_y), color=(0.85, 0.88, 0.92))
        page2.insert_text((50, row_y + 14), dim, fontsize=8.5, fontname="helv", color=(0.2, 0.2, 0.2))
        page2.insert_text((270, row_y + 14), val, fontsize=8.5, fontname="helv", color=(0.1, 0.1, 0.1))
        page2.insert_text((410, row_y + 14), scp, fontsize=8.5, fontname="helv", color=(0.3, 0.4, 0.5))

    # Footer Page 2
    page2.draw_line((40, 800), (555, 800), color=(0.8, 0.8, 0.8))
    page2.insert_text((40, 815), "NovaCorp Technologies Limited | FY2024 Review", fontsize=8, fontname="helv", color=(0.5, 0.5, 0.5))
    page2.insert_text((515, 815), "Page 2 of 2", fontsize=8, fontname="helv", color=(0.5, 0.5, 0.5))

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()
    print(f"Successfully generated sample PDF at: {output_path}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "sample-test.pdf"
    generate_sample_pdf(out)
