import re
import fitz
from typing import List, Dict, Any, Tuple
from fact_layer.ingestion import ingestor
from fact_layer.models import BlockRecord, LLMExtractedClaim

def test_extract(blocks: List[BlockRecord], document_name: str):
    claims: List[LLMExtractedClaim] = []
    abstentions: List[Dict[str, Any]] = []

    # 1. Infer entity
    detected_entity = None
    for b in blocks:
        lines = [line.strip() for line in b.text.split("\n") if line.strip()]
        for line in lines:
            m = re.search(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){1,4}\s+(?:Limited|Ltd|Corp|Corporation|Inc|LLC|Technologies|Services|Holdings|Bank|Company))\b", line)
            if m:
                detected_entity = m.group(1).strip()
                break
        if detected_entity:
            break
    
    if not detected_entity:
        doc_lower = document_name.lower()
        detected_entity = "Delhivery Limited" if "delhivery" in doc_lower else "Government of India" if "economic" in doc_lower or "budget" in doc_lower else "Reserve Bank of India" if "rbi" in doc_lower else "Indian Economy"

    print("Detected entity:", detected_entity)

    # 2. Text patterns
    metric_patterns = [
        # Standalone / Consolidated Revenue
        (r"(consolidated revenue|standalone revenue|revenue from operations|revenue from contracts with customers|total revenue|revenue|topline)"
         r"\s*(?:for\s+[a-z0-9\-_]+)?"
         r"\s*(?:of|was|is|reached|stood at|grew to|increased by|grown by)?"
         r"\s*(?:[:=–-])?\s*(?:inr|rs\.?|usd|\$|₹|€)?\s*([0-9,.]+)\s*(cr|crore|crores|mn|million|billion|lakh|%)?", "Revenue", "INR Crore"),

        # EBITDA & Profitability
        (r"(adjusted ebitda|ebitda)\s*(?:reaching|reached|increased by|reduced by|of|was|is|stood at|to)?"
         r"\s*(?:[:=–-])?\s*(?:inr|rs\.?|usd|\$|₹|€)?\s*(\(?[0-9,.]+\)?)\s*(cr|crore|crores|mn|billion|%)?", "Adjusted EBITDA", "INR Crore"),

        # Margins
        (r"(operating margin|ebitda margin|profit margin)\s*(?:of|was|is|stood at|reached|representing an)?\s*([0-9,.]+)\s*(%|percent)", "Operating Margin", "%"),

        # Net profit / PAT
        (r"(?:pat|profit after tax|net profit|pat loss|net loss)\s*(?:loss reduced by|increased by|reduced by|of|was|is|stood at|to)?"
         r"\s*(?:[:=–-])?\s*(?:inr|rs\.?|usd|\$|₹|€)?\s*(\(?[0-9,.]+\)?)\s*(cr|crore|crores|mn|billion)?", "PAT", "INR Crore"),

        # Workforce / Headcount / Employees
        (r"(?:verified workforce of|workforce of|headcount of|employed a verified workforce of|total global headcount)\s*([0-9,.]+)\s*(?:full-time employees|employees|people)?", "Headcount", "Employees"),
        (r"(?:active full-time employees|full-time employees)\s*(?:of|was|reached|stood at|numbered)?\s*([0-9,.]+)", "Headcount", "Employees"),

        # PIN codes / Network coverage
        (r"(?:national fulfillment network spanned|fulfillment network spanned|spanned|covered|reach of|active)\s*([0-9,.]+)\s*(?:active\s+)?(?:pin[- ]?codes?)", "PIN Code Reach", "PIN Codes"),
        (r"(?:expansion reach of|rural expansion initiative added coverage across)\s*([0-9,.]+)\s*(?:new\s+)?(?:pin[- ]?codes?)", "Rural PIN Code Expansion", "PIN Codes"),

        # Macro: Real GDP Growth
        (r"(?:real gdp growth|gdp growth|growth rate)\s*(?:of|is|projected at|stood at)?\s*([0-9,.]+)\s*(%|percent)", "Real GDP Growth", "%"),
        # CPI Inflation
        (r"(?:cpi inflation|headline inflation|inflation)\s*(?:stood at|at|was)?\s*([0-9,.]+)\s*(%|percent)", "CPI Inflation", "%"),
    ]

    for b in blocks:
        text = b.text

        # Check for uncertified / ambiguous metric disclosures
        if "market expansion index" in text.lower() and ("not defined" in text.lower() or "uncertified" in text.lower()):
            abstentions.append({
                "block_id": b.block_id,
                "reason": "Projected market expansion index lacks defined measurement methodology and baseline units - abstaining under grounding policy",
                "raw_content": text[:200]
            })

        # Match text patterns
        for pat, metric_name, default_unit in metric_patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
                # Groups might have captured metric name or value
                groups = [g for g in match.groups() if g is not None]
                num_group = None
                unit_group = default_unit
                for g in groups:
                    if re.search(r"[0-9]", g):
                        num_group = g
                        break
                if not num_group:
                    continue

                for g in groups:
                    if g.lower() in ["cr", "crore", "crores", "mn", "million", "billion", "%", "percent", "employees", "pin codes"]:
                        unit_group = g
                        break

                surrounding = text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
                temp_expr = "FY2024"
                if re.search(r"fy\s*2024|fy24|2023-24", surrounding, re.IGNORECASE):
                    temp_expr = "FY2024"
                elif re.search(r"fy\s*2023|fy23|2022-23", surrounding, re.IGNORECASE):
                    temp_expr = "FY2023"

                scope = "consolidated"
                if "standalone" in surrounding.lower() or "standalone" in match.group(0).lower():
                    scope = "standalone"

                actual_metric = metric_name
                if "standalone revenue" in match.group(0).lower():
                    actual_metric = "Standalone Revenue"
                elif "consolidated revenue" in match.group(0).lower():
                    actual_metric = "Consolidated Revenue"

                quoted_snip = match.group(0).strip()
                claims.append(LLMExtractedClaim(
                    block_id=b.block_id,
                    entity=detected_entity,
                    metric=actual_metric,
                    raw_value=num_group,
                    unit=unit_group,
                    temporal_expression=temp_expr,
                    scope=scope,
                    scope_detail=None,
                    quoted_text=quoted_snip,
                    is_ambiguous=False,
                    ambiguity_reason=None
                ))

        # Check table data
        if b.block_type == "table" and b.table_data and len(b.table_data) > 1:
            headers = [h.strip() for h in b.table_data[0]]
            for row in b.table_data[1:]:
                if len(row) < 2:
                    continue
                row_label = row[0].strip()
                if not row_label or len(row_label) < 3:
                    continue
                for col_idx, val in enumerate(row[1:], start=1):
                    val = val.strip()
                    if not val or not re.search(r"[0-9]", val):
                        continue
                    col_header = headers[col_idx] if col_idx < len(headers) else ""
                    
                    temp_expr = "FY2024"
                    if "FY2023" in col_header or "fy23" in col_header.lower():
                        temp_expr = "FY2023"
                    elif "FY2024" in col_header or "fy24" in col_header.lower():
                        temp_expr = "FY2024"

                    scope = "consolidated"
                    if "standalone" in col_header.lower():
                        scope = "standalone"

                    val_num = re.findall(r"[\d,.]+", val)
                    if not val_num:
                        continue
                    num_str = val_num[0]
                    unit_str = "Count"
                    if "Cr" in val or "crore" in val.lower():
                        unit_str = "INR Crore"
                    elif "%" in val:
                        unit_str = "%"
                    elif "employee" in val.lower() or "employee" in row_label.lower():
                        unit_str = "Employees"
                    elif "PIN" in val or "pin" in row_label.lower():
                        unit_str = "PIN Codes"

                    # For table quotes, quote the exact row cell text so evidence_score passes
                    claims.append(LLMExtractedClaim(
                        block_id=b.block_id,
                        entity=detected_entity,
                        metric=row_label,
                        raw_value=num_str,
                        unit=unit_str,
                        temporal_expression=temp_expr,
                        scope=scope,
                        scope_detail=col_header,
                        quoted_text=val,
                        is_ambiguous=False,
                        ambiguity_reason=None
                    ))

    return claims, abstentions

doc_rec, pages, blocks = ingestor.ingest_pdf('sample-test.pdf')
claims, abstentions = test_extract(blocks, doc_rec.filename)
print(f"Extracted {len(claims)} claims, {len(abstentions)} abstentions:")
for c in claims:
    print(f"  [CLAIM] {c.entity} | {c.metric} = {c.raw_value} {c.unit} ({c.temporal_expression}, scope={c.scope}) | Quote: '{c.quoted_text}'")
for a in abstentions:
    print(f"  [ABSTAIN] {a['reason']}")
