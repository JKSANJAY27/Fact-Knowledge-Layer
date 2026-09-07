from __future__ import annotations
import json
import re
import time
import difflib
from typing import List, Dict, Any, Optional, Tuple
import requests

from fact_layer.config import GEMINI_API_KEY, GEMINI_MODEL
from fact_layer.models import (
    BlockRecord,
    AtomicFact,
    SourcePointer,
    ExtractionFailure,
    LLMExtractedClaim,
    LLMExtractionBatch,
    FailureType
)
from fact_layer.normalization import (
    normalize_unit_and_value,
    normalize_temporal_expression,
    parse_numeric_value
)
from fact_layer.storage import storage


EXTRACTION_SYSTEM_PROMPT = """You are an expert financial and institutional fact extraction engine.
Given a list of document blocks with assigned 'block_id's, your task is to extract ATOMIC FINANCIAL & INSTITUTIONAL CLAIMS.

RULES:
1. One claim = one subject (entity), one predicate (metric), and one value.
2. The 'block_id' MUST match one of the block_ids provided in the input. NEVER invent block IDs or page numbers.
3. 'quoted_text' MUST be an exact verbatim substring from the corresponding block text.
4. If a block has numbers but the unit is missing, ambiguous, or unstated, DO NOT GUESS! Either report it with unit='Unknown' and is_ambiguous=true, or add an abstention entry in 'abstentions'.
5. If the temporal period or scope is ambiguous (e.g., cannot determine if FY23 or FY24, or Consolidated vs Standalone), mark is_ambiguous=true with the ambiguity_reason.
6. Return a valid JSON object matching the schema below.

JSON Schema:
{
  "claims": [
    {
      "block_id": "blk_...",
      "entity": "Delhivery Limited",
      "metric": "Revenue from contracts with customers",
      "raw_value": "8,142.17",
      "unit": "INR Crore",
      "temporal_expression": "FY2024",
      "scope": "consolidated",
      "scope_detail": null,
      "quoted_text": "Revenue from contracts with customers ₹8,142.17 Cr",
      "is_ambiguous": false,
      "ambiguity_reason": null
    }
  ],
  "abstentions": [
    {
      "block_id": "blk_...",
      "reason": "Missing unit on table column 2",
      "raw_content": "snippet..."
    }
  ]
}
"""


class FactExtractor:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model or GEMINI_MODEL
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    def _call_gemini_api(self, prompt: str, max_retries: int = 1) -> Optional[str]:
        """Calls Gemini API with fast fallback and JSON mode"""
        if not self.api_key:
            return None

        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": EXTRACTION_SYSTEM_PROMPT},
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "response_mime_type": "application/json"
            }
        }

        for attempt in range(max_retries):
            try:
                res = requests.post(self.endpoint, headers=headers, json=payload, timeout=20)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        content_parts = candidates[0].get("content", {}).get("parts", [])
                        if content_parts:
                            return content_parts[0].get("text", "")
            except Exception:
                break

        return None

    def extract_facts_from_blocks(
        self,
        blocks: List[BlockRecord],
        document_id: str,
        document_name: str,
        page_number: int,
        use_llm: bool = True
    ) -> Tuple[List[AtomicFact], List[ExtractionFailure]]:
        """
        Extracts atomic facts from a set of blocks from a single page.
        Validates quotes server-side against the raw text of the assigned block.
        Calculates confidence score heuristically.
        Explicitly logs any ambiguities/abstentions into the failures table.
        """
        if not blocks:
            return [], []

        # Filter blocks that have meaningful content
        content_blocks = [b for b in blocks if len(b.text.strip()) > 10]
        if not content_blocks:
            return [], []

        # Prepare block payload for the LLM
        blocks_payload = []
        block_map: Dict[str, BlockRecord] = {}
        for b in content_blocks:
            block_map[b.block_id] = b
            blocks_payload.append({
                "block_id": b.block_id,
                "block_type": b.block_type,
                "text": b.text[:1200]  # truncate huge blocks to avoid token limits
            })

        extracted_claims: List[LLMExtractedClaim] = []
        raw_abstentions: List[Dict[str, Any]] = []

        if use_llm and self.api_key:
            prompt = f"""DOCUMENT: {document_name} | PAGE: {page_number}
BLOCKS TO EXTRACT:
{json.dumps(blocks_payload, indent=2)}

Extract atomic claims and record any abstentions according to the instructions. Return JSON only."""

            llm_raw_response = self._call_gemini_api(prompt)
            if llm_raw_response:
                try:
                    parsed_json = json.loads(llm_raw_response)
                    claims_data = parsed_json.get("claims", [])
                    raw_abstentions = parsed_json.get("abstentions", [])
                    for cd in claims_data:
                        extracted_claims.append(LLMExtractedClaim(**cd))
                except Exception:
                    extracted_claims, raw_abstentions = self._fallback_rule_based_extract(content_blocks, document_name)
            else:
                extracted_claims, raw_abstentions = self._fallback_rule_based_extract(content_blocks, document_name)
        else:
            extracted_claims, raw_abstentions = self._fallback_rule_based_extract(content_blocks, document_name)

        validated_facts: List[AtomicFact] = []
        failures: List[ExtractionFailure] = []

        # Process abstentions from LLM
        for abst in raw_abstentions:
            b_id = abst.get("block_id", content_blocks[0].block_id)
            failures.append(ExtractionFailure(
                document_id=document_id,
                document_name=document_name,
                page_number=page_number,
                block_id=b_id,
                failure_type="EXTRACTION_ABSTAINED",
                reason=abst.get("reason", "Extraction abstained due to ambiguity or missing metadata"),
                raw_content=abst.get("raw_content", "")
            ))

        # Process extracted claims
        for claim in extracted_claims:
            # 1. Server-side block_id resolution
            target_block = block_map.get(claim.block_id)
            if not target_block:
                # LLM hallucinated a block_id! Log failure under abstention principle
                failures.append(ExtractionFailure(
                    document_id=document_id,
                    document_name=document_name,
                    page_number=page_number,
                    block_id=claim.block_id,
                    failure_type="UNGROUNDED_CLAIM",
                    reason=f"LLM produced non-existent block_id '{claim.block_id}' - rejected under strict provenance rule",
                    raw_content=f"{claim.entity} | {claim.metric} = {claim.raw_value}"
                ))
                continue

            # 2. Check for explicit ambiguity or missing unit
            if claim.is_ambiguous or claim.unit.lower() in ["unknown", "none", "missing", ""]:
                f_type: FailureType = "MISSING_UNIT" if claim.unit.lower() in ["unknown", "none", "missing", ""] else "AMBIGUOUS_METRIC"
                failures.append(ExtractionFailure(
                    document_id=document_id,
                    document_name=document_name,
                    page_number=page_number,
                    block_id=claim.block_id,
                    failure_type=f_type,
                    reason=claim.ambiguity_reason or f"Missing or ambiguous unit/metric for value '{claim.raw_value}'",
                    raw_content=claim.quoted_text or target_block.text[:200]
                ))
                # Do not emit ungrounded or ambiguous facts - principle of abstention over wrong answer
                continue

            # 3. Grounding & Evidence verification: check quoted_text in block text
            evidence_score = 0.0
            block_text_lower = target_block.text.lower()
            quote_lower = claim.quoted_text.lower().strip()

            if quote_lower and quote_lower in block_text_lower:
                evidence_score = 1.0
            elif claim.raw_value.lower() in block_text_lower:
                # Raw value appears in block, compute sequence matcher on quote
                matcher = difflib.SequenceMatcher(None, quote_lower, block_text_lower)
                evidence_score = max(0.5, matcher.quick_ratio())
            else:
                # Quoted text doesn't match block content!
                failures.append(ExtractionFailure(
                    document_id=document_id,
                    document_name=document_name,
                    page_number=page_number,
                    block_id=claim.block_id,
                    failure_type="UNGROUNDED_CLAIM",
                    reason=f"Quoted evidence '{claim.quoted_text}' not verified in target block text",
                    raw_content=target_block.text[:200]
                ))
                continue

            # 4. Normalization
            norm_val, canon_unit = normalize_unit_and_value(claim.raw_value, claim.unit)
            norm_temporal = normalize_temporal_expression(claim.temporal_expression)

            norm_score = 1.0 if norm_val is not None and norm_temporal.canonical_period != "Unknown" else 0.5
            clarity_score = 1.0 if len(claim.entity) > 2 and len(claim.metric) > 3 else 0.6

            # Weighted confidence heuristic: clarity (35%) + evidence (35%) + normalization (30%)
            overall_confidence = round(
                (0.35 * clarity_score) + (0.35 * evidence_score) + (0.30 * norm_score),
                3
            )

            source_ptr = SourcePointer(
                document_id=document_id,
                document_name=document_name,
                page_number=page_number,
                block_id=claim.block_id,
                quoted_text=claim.quoted_text,
                bbox=target_block.bbox
            )

            fact = AtomicFact(
                document_id=document_id,
                page_number=page_number,
                block_id=claim.block_id,
                entity=claim.entity.strip(),
                metric=claim.metric.strip(),
                raw_value=claim.raw_value.strip(),
                unit=claim.unit.strip(),
                normalized_value=norm_val,
                canonical_unit=canon_unit,
                temporal_context=norm_temporal,
                scope=claim.scope,
                scope_detail=claim.scope_detail,
                source_pointer=source_ptr,
                confidence_score=overall_confidence,
                confidence_breakdown={
                    "clarity": clarity_score,
                    "evidence_match": round(evidence_score, 2),
                    "normalization": norm_score
                }
            )
            validated_facts.append(fact)

        # Persist extracted facts and failures
        storage.add_facts(validated_facts)
        storage.add_failures(failures)

        return validated_facts, failures

    def _fallback_rule_based_extract(
        self,
        blocks: List[BlockRecord],
        document_name: str
    ) -> Tuple[List[LLMExtractedClaim], List[Dict[str, Any]]]:
        """
        Deterministic rule-based extractor that runs if LLM is unavailable or for rapid local testing.
        Extracts key financial and institutional patterns.
        """
        claims: List[LLMExtractedClaim] = []
        abstentions: List[Dict[str, Any]] = []

        # Determine default entity from document name
        doc_lower = document_name.lower()
        default_entity = "Delhivery Limited" if "delhivery" in doc_lower else "Government of India" if "economic" in doc_lower or "budget" in doc_lower else "Reserve Bank of India" if "rbi" in doc_lower else "Indian Economy"

        # Regex patterns for financial and operational metrics
        metric_patterns = [
            # Revenue from contracts
            (r"(?:revenue from contracts with customers|total revenue|revenue|topline)\s*(?:of|was|is|reached|increased by|grown by)?\s*(?:[:=–-]?)\s*(?:[₹$€]|rs\.?\s*)?([0-9,.]+)\s*(cr|crore|crores|mn|million|billion|lakh|%)?", "Revenue", "INR Crore"),
            # EBITDA & Profitability
            (r"(?:adjusted ebitda|ebitda)\s*(?:increased by|reduced by|of|was|is|stood at|to)?\s*(?:[:=–-]?)\s*(?:[₹$€]|rs\.?\s*)?(\(?[0-9,.]+\)?)\s*(cr|crore|crores|mn|billion|%)?", "Adjusted EBITDA", "INR Crore"),
            (r"(?:from\s*(?:rs\.?\s*)?(\(?[0-9,.]+\)?)\s*(cr|crore|crores)?\s*in\s*(fy\s*[0-9]{2,4}))", "Adjusted EBITDA", "INR Crore"),
            # PAT (Profit after tax) / Net Profit / Loss
            (r"(?:pat|profit after tax|net profit|pat loss|net loss)\s*(?:loss reduced by|increased by|reduced by|of|was|is|stood at|to)?\s*(?:[:=–-]?)\s*(?:[₹$€]|rs\.?\s*)?(\(?[0-9,.]+\)?)\s*(cr|crore|crores|mn|billion)?", "PAT", "INR Crore"),
            # Segment Growth / Service Profitability
            (r"(?:ptl|tl|scs|express parcel)\s*:\s*([0-9,.]+)\s*%\+?\s*(?:yoy\s*growth|service ebitda|revenue growth)?", "Segment Growth", "%"),
            # Real GDP Growth
            (r"(?:real gdp growth|gdp growth|growth rate)\s*(?:of|is|projected at|stood at)?\s*([0-9,.]+)\s*(%|percent)", "Real GDP Growth", "%"),
            # Express Parcel Volumes
            (r"(?:express parcel volume|shipment volume|parcel volume|express parcel)\s*(?:of|was|reached)?\s*([0-9,.]+)\s*(million|mn|cr|crore|packages|tonnes)?", "Express Parcel Volume", "Million Packages"),
            # Pin code reach
            (r"(?:pin[- ]?code reach|pin[- ]?codes)\s*(?:covered|reached|of)?\s*([0-9,.]+)", "Pin Code Reach", "Pin Codes"),
            # Network service points / facilities
            (r"(?:gateways|automated sort centers|facilities|fulfillment centers)\s*(?:of|numbered)?\s*([0-9,.]+)", "Network Facilities", "Count"),
            # Working Capital
            (r"(?:nwc days|working capital days)\s*(?:from\s*[0-9,.]+\s*to\s*)?([0-9,.]+)\s*(?:days)?", "NWC Days", "Days"),
            # Inflation
            (r"(?:cpi inflation|headline inflation|inflation)\s*(?:stood at|at|was)?\s*([0-9,.]+)\s*(%|percent)", "CPI Inflation", "%"),
            # Fiscal Deficit
            (r"(?:fiscal deficit)\s*(?:of|was|is|stood at)?\s*([0-9,.]+)\s*(%|percent|cr|crore)?", "Fiscal Deficit", "%"),
            # Foreign Exchange Reserves
            (r"(?:foreign exchange reserves|forex reserves)\s*(?:stood at|of|were)?\s*(?:[₹$€]|rs\.?\s*)?([0-9,.]+)\s*(billion|million|bn|mn)?", "Foreign Exchange Reserves", "USD Billion"),
        ]

        for b in blocks:
            text = b.text

            # Check for ambiguous lines without unit on tables -> log as abstention
            if b.block_type == "table" and b.table_data:
                headers = b.table_data[0] if len(b.table_data) > 0 else []
                header_text = " ".join(headers).lower()
                if not any(u in header_text for u in ["cr", "inr", "rs", "percent", "%", "lakh", "crore", "usd", "$"]):
                    abstentions.append({
                        "block_id": b.block_id,
                        "reason": f"Table on page {b.page_number} lacks explicit column unit indicators - abstaining from ungrounded claims",
                        "raw_content": text[:150]
                    })

            # Check text against metric patterns
            for pat, metric_name, default_unit in metric_patterns:
                for match in re.finditer(pat, text, re.IGNORECASE):
                    val_str = match.group(1)
                    if not val_str or not re.search(r"[0-9]", val_str):
                        continue

                    unit_str = default_unit
                    if len(match.groups()) >= 2 and match.group(2):
                        unit_str = match.group(2)

                    # Determine temporal context from sentence or block
                    surrounding = text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
                    temp_expr = "FY2024"
                    if re.search(r"fy\s*2024|fy24|2023-24", surrounding, re.IGNORECASE):
                        temp_expr = "FY2024"
                    elif re.search(r"fy\s*2023|fy23|2022-23", surrounding, re.IGNORECASE):
                        temp_expr = "FY2023"
                    elif re.search(r"fy\s*2022|fy22|2021-22", surrounding, re.IGNORECASE):
                        temp_expr = "FY2022"
                    elif re.search(r"q4\s*fy24|q4fy24", surrounding, re.IGNORECASE):
                        temp_expr = "Q4_FY2024"
                    elif re.search(r"q3\s*fy24|q3fy24", surrounding, re.IGNORECASE):
                        temp_expr = "Q3_FY2024"
                    elif re.search(r"2024-25|fy25", surrounding, re.IGNORECASE):
                        temp_expr = "FY2025"

                    scope = "consolidated"
                    if "standalone" in surrounding.lower():
                        scope = "standalone"
                    elif any(s in text.lower() for s in ["ptl", "tl", "scs", "express parcel"]):
                        scope = "segment"

                    # Segment specific metric
                    actual_metric = metric_name
                    if "ptl" in surrounding.lower():
                        actual_metric = f"PTL {metric_name}" if "segment" in metric_name.lower() else metric_name
                    elif "express parcel" in surrounding.lower():
                        actual_metric = f"Express Parcel {metric_name}" if "segment" in metric_name.lower() else metric_name

                    quoted_snip = match.group(0).strip()

                    claims.append(LLMExtractedClaim(
                        block_id=b.block_id,
                        entity=default_entity,
                        metric=actual_metric,
                        raw_value=val_str,
                        unit=unit_str,
                        temporal_expression=temp_expr,
                        scope=scope,
                        scope_detail=None,
                        quoted_text=quoted_snip,
                        is_ambiguous=False,
                        ambiguity_reason=None
                    ))

        return claims, abstentions


# Singleton instance
extractor = FactExtractor()
