from __future__ import annotations
import difflib
import json
import re
from typing import List, Dict, Any, Optional, Tuple
import requests

from fact_layer.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    VALUE_TOLERANCE_PERCENT,
    RECONCILIATION_SIMILARITY_THRESHOLD
)
from fact_layer.models import (
    AtomicFact,
    FactRelationship,
    RelationshipType
)
from fact_layer.storage import storage


def canonical_entity_name(entity: str) -> str:
    """Standardizes company and institutional names"""
    e = entity.strip().lower()
    e = re.sub(r"\b(limited|ltd|inc|corp|corporation|pvt|private)\b", "", e)
    e = re.sub(r"[^a-z0-9]", "", e)
    return e


def metric_similarity(m1: str, m2: str) -> float:
    """Computes semantic/token similarity between two metric names"""
    m1_clean = m1.strip().lower()
    m2_clean = m2.strip().lower()
    if m1_clean == m2_clean:
        return 1.0

    # Common financial metric equivalences
    equivalences = [
        {"revenue", "total revenue", "revenue from contracts with customers", "topline"},
        {"adjusted ebitda", "ebitda", "operating profit before depreciation"},
        {"pat", "profit after tax", "net profit", "pat loss", "net loss"},
        {"real gdp growth", "gdp growth", "real gdp growth rate"},
        {"cpi inflation", "headline inflation", "inflation", "cpi-c inflation"},
        {"express parcel volume", "parcel volume", "shipment volume"},
        {"pin code reach", "pin codes", "pincodes covered"},
    ]

    for eq_set in equivalences:
        if any(term in m1_clean for term in eq_set) and any(term in m2_clean for term in eq_set):
            return 0.95

    return difflib.SequenceMatcher(None, m1_clean, m2_clean).ratio()


class ReconciliationEngine:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model or GEMINI_MODEL
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    def reconcile_facts(self, facts: Optional[List[AtomicFact]] = None) -> List[FactRelationship]:
        """
        Runs the multi-stage reconciliation pipeline over candidate fact pairs:
        1. Candidate pair filter: same entity + metric similarity >= threshold
        2. Deterministic 6-step cascade:
           Entity -> Metric -> Unit -> Period -> Scope -> Value
        3. Ambiguity adjudication via LLM with structured schema
        """
        if facts is None:
            facts = storage.list_facts()

        if len(facts) < 2:
            return []

        relationships: List[FactRelationship] = []

        # Find candidate pairs (fact_a, fact_b) where a.fact_id < b.fact_id
        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                fact_a = facts[i]
                fact_b = facts[j]

                # Skip exact duplicate extractions from the identical block and value
                if fact_a.document_id == fact_b.document_id and fact_a.block_id == fact_b.block_id and fact_a.raw_value == fact_b.raw_value:
                    continue

                # Step 1: Candidate pair filter (same entity + similar metric)
                ent_a = canonical_entity_name(fact_a.entity)
                ent_b = canonical_entity_name(fact_b.entity)

                # Entity match check
                if ent_a != ent_b and not (ent_a in ent_b or ent_b in ent_a):
                    continue

                # Metric similarity check
                sim = metric_similarity(fact_a.metric, fact_b.metric)
                if sim < RECONCILIATION_SIMILARITY_THRESHOLD:
                    continue

                # Run deterministic cascade
                rel = self._evaluate_pair(fact_a, fact_b, sim)
                if rel:
                    relationships.append(rel)

        # Save relationships to storage
        storage.add_relationships(relationships)
        return relationships

    def _evaluate_pair(self, a: AtomicFact, b: AtomicFact, metric_sim: float) -> Optional[FactRelationship]:
        """
        Applies deterministic checks in strict order:
        same entity -> same metric -> same unit -> same period -> same scope -> values equal?
        """
        # 1. Unit compatibility check
        if a.canonical_unit != b.canonical_unit:
            # Check if one is None or unknown
            if not a.canonical_unit or not b.canonical_unit or a.canonical_unit == "Unknown" or b.canonical_unit == "Unknown":
                return FactRelationship(
                    fact_a_id=a.fact_id,
                    fact_b_id=b.fact_id,
                    relation_type="UNRESOLVED",
                    confidence=0.5,
                    explanation=f"Cannot resolve relationship: Missing or unknown unit in comparison ({a.unit} vs {b.unit}).",
                    difference_field="unit",
                    fact_a=a,
                    fact_b=b
                )
            # Incompatible units (e.g. INR vs USD, or Count vs %)
            return FactRelationship(
                fact_a_id=a.fact_id,
                fact_b_id=b.fact_id,
                relation_type="CONTEXTUAL_DIFFERENCE",
                confidence=0.9,
                explanation=f"Differs in measurement unit: Fact A is reported in {a.unit} ({a.raw_value}) while Fact B is in {b.unit} ({b.raw_value}).",
                difference_field="unit",
                fact_a=a,
                fact_b=b
            )

        # 2. Period check
        period_a = a.temporal_context.canonical_period
        period_b = b.temporal_context.canonical_period

        if period_a != "Unknown" and period_b != "Unknown" and period_a != period_b:
            return FactRelationship(
                fact_a_id=a.fact_id,
                fact_b_id=b.fact_id,
                relation_type="CONTEXTUAL_DIFFERENCE",
                confidence=0.95,
                explanation=(
                    f"Differs in temporal period: Fact A from {a.source_pointer.document_name} "
                    f"pertains to {period_a} ({a.raw_value} {a.unit}), whereas Fact B from "
                    f"{b.source_pointer.document_name} pertains to {period_b} ({b.raw_value} {b.unit})."
                ),
                difference_field="period",
                fact_a=a,
                fact_b=b
            )

        # 3. Scope check
        if a.scope != b.scope:
            return FactRelationship(
                fact_a_id=a.fact_id,
                fact_b_id=b.fact_id,
                relation_type="CONTEXTUAL_DIFFERENCE",
                confidence=0.92,
                explanation=(
                    f"Differs in disclosure scope: Fact A is reported under '{a.scope}' scope ({a.raw_value} {a.unit}), "
                    f"while Fact B is reported under '{b.scope}' scope ({b.raw_value} {b.unit})."
                ),
                difference_field="scope",
                fact_a=a,
                fact_b=b
            )

        # 4. Values check (entity, metric, unit, period, and scope all match!)
        if a.normalized_value is not None and b.normalized_value is not None:
            diff = abs(a.normalized_value - b.normalized_value)
            denom = max(abs(a.normalized_value), abs(b.normalized_value), 1e-6)
            rel_diff = diff / denom

            if rel_diff <= VALUE_TOLERANCE_PERCENT:
                # CORROBORATED
                return FactRelationship(
                    fact_a_id=a.fact_id,
                    fact_b_id=b.fact_id,
                    relation_type="CORROBORATED",
                    confidence=round(min(a.confidence_score, b.confidence_score) * 0.98, 2),
                    explanation=(
                        f"Corroborated across disclosures: Both {a.source_pointer.document_name} (page {a.source_pointer.page_number}) "
                        f"and {b.source_pointer.document_name} (page {b.source_pointer.page_number}) agree on "
                        f"{a.entity} {a.metric} as {a.raw_value} {a.unit} for {period_a} ({a.scope})."
                    ),
                    difference_field=None,
                    fact_a=a,
                    fact_b=b
                )
            else:
                # CONTRADICTED (labeled as "likely contradiction," never certain)
                return FactRelationship(
                    fact_a_id=a.fact_id,
                    fact_b_id=b.fact_id,
                    relation_type="CONTRADICTED",
                    confidence=0.88,
                    explanation=(
                        f"Likely contradiction: Both sources report {a.entity} {a.metric} for the same period "
                        f"({period_a}) and scope ({a.scope}), but report conflicting values: "
                        f"{a.raw_value} in {a.source_pointer.document_name} vs "
                        f"{b.raw_value} in {b.source_pointer.document_name}."
                    ),
                    difference_field="value",
                    fact_a=a,
                    fact_b=b
                )

        # 5. Genuinely ambiguous pair: invoke LLM adjudication if available, else UNRESOLVED
        return self._adjudicate_ambiguous_pair(a, b)

    def _adjudicate_ambiguous_pair(self, a: AtomicFact, b: AtomicFact) -> FactRelationship:
        """Adjudicates ambiguous candidate pairs using structured JSON LLM output"""
        prompt = f"""Compare two extracted financial/institutional facts:
Fact A:
  Document: {a.source_pointer.document_name} (Page {a.source_pointer.page_number})
  Entity: {a.entity}
  Metric: {a.metric}
  Raw Value: {a.raw_value} ({a.unit})
  Period: {a.temporal_context.canonical_period}
  Scope: {a.scope}
  Evidence Quote: "{a.source_pointer.quoted_text}"

Fact B:
  Document: {b.source_pointer.document_name} (Page {b.source_pointer.page_number})
  Entity: {b.entity}
  Metric: {b.metric}
  Raw Value: {b.raw_value} ({b.unit})
  Period: {b.temporal_context.canonical_period}
  Scope: {b.scope}
  Evidence Quote: "{b.source_pointer.quoted_text}"

Determine the exact relationship: CORROBORATED, CONTRADICTED, CONTEXTUAL_DIFFERENCE, or UNRESOLVED.
Provide a concise 1-2 sentence factual explanation. Return JSON:
{{
  "relation_type": "CORROBORATED|CONTRADICTED|CONTEXTUAL_DIFFERENCE|UNRESOLVED",
  "explanation": "...",
  "difference_field": "period|scope|value|metric|unit|null"
}}"""

        if self.api_key:
            headers = {"Content-Type": "application/json", "X-goog-api-key": self.api_key}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0, "response_mime_type": "application/json"}
            }
            try:
                res = requests.post(self.endpoint, headers=headers, json=payload, timeout=12)
                if res.status_code == 200:
                    cand = res.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")
                    if cand:
                        parsed = json.loads(cand)
                        rtype = parsed.get("relation_type", "UNRESOLVED")
                        return FactRelationship(
                            fact_a_id=a.fact_id,
                            fact_b_id=b.fact_id,
                            relation_type=rtype if rtype in ["CORROBORATED", "CONTRADICTED", "CONTEXTUAL_DIFFERENCE", "UNRESOLVED"] else "UNRESOLVED",
                            confidence=0.82,
                            explanation=parsed.get("explanation", "LLM adjudicated relationship."),
                            difference_field=parsed.get("difference_field"),
                            fact_a=a,
                            fact_b=b
                        )
            except Exception:
                pass

        # Fallback to UNRESOLVED under abstention principle
        return FactRelationship(
            fact_a_id=a.fact_id,
            fact_b_id=b.fact_id,
            relation_type="UNRESOLVED",
            confidence=0.5,
            explanation=f"Ambiguous comparison between '{a.metric}' and '{b.metric}': insufficient structured metadata to determine agreement.",
            difference_field=None,
            fact_a=a,
            fact_b=b
        )


# Singleton instance
reconciler = ReconciliationEngine()
