from __future__ import annotations
import difflib
import json
import re
from typing import List, Dict, Any, Optional
import requests

from fact_layer.config import GEMINI_API_KEY, GEMINI_MODEL
from fact_layer.models import (
    AtomicFact,
    QueryRequest,
    QueryResponse,
    GroundedCitation
)
from fact_layer.storage import storage


class FactRetriever:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model or GEMINI_MODEL
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    def retrieve_relevant_facts(
        self,
        query: str,
        entity_filter: Optional[str] = None,
        period_filter: Optional[str] = None,
        top_k: int = 10
    ) -> List[AtomicFact]:
        """
        Retrieves relevant atomic facts using hybrid search:
        - Structured filters (entity, period)
        - Token/keyword overlap and sequence similarity against entity, metric, and quotes
        - Weighted by fact confidence score
        """
        all_facts = storage.list_facts()
        if not all_facts:
            return []

        scored_facts = []
        STOPWORDS = {"what", "was", "is", "are", "were", "the", "in", "of", "at", "to", "for", "from", "and", "a", "an", "by", "on", "with", "as", "about", "how", "much", "many", "does", "did"}
        q_tokens = {t for t in re.findall(r"\w+", query.lower()) if t not in STOPWORDS and len(t) > 1}

        for fact in all_facts:
            # Apply hard filters if provided
            if entity_filter and entity_filter.lower() not in fact.entity.lower():
                continue
            if period_filter and period_filter.lower() != fact.temporal_context.canonical_period.lower():
                continue

            # Compute match score
            fact_tokens = {
                t for t in re.findall(
                    r"\w+",
                    f"{fact.entity} {fact.metric} {fact.temporal_context.canonical_period} {fact.scope} {fact.source_pointer.quoted_text}".lower()
                ) if t not in STOPWORDS and len(t) > 1
            }

            overlap = len(q_tokens.intersection(fact_tokens))
            seq_score = difflib.SequenceMatcher(None, query.lower(), f"{fact.entity} {fact.metric}".lower()).ratio()

            # Require either token overlap or strong sequence similarity
            if overlap == 0 and seq_score < 0.35:
                continue

            score = (overlap * 2.5) + (seq_score * 3.0) + (fact.confidence_score * 1.0)
            scored_facts.append((score, fact))

        scored_facts.sort(key=lambda x: x[0], reverse=True)
        return [f for score, f in scored_facts[:top_k] if score > 0.5]

    def answer_query(self, request: QueryRequest) -> QueryResponse:
        """
        Answers a user query strictly using retrieved structured facts and known relationships:
        - Never invents numbers
        - Cites document provenance and page numbers
        - Highlights any corroborations or contradictions in the evidence
        """
        retrieved_facts = self.retrieve_relevant_facts(
            query=request.query,
            entity_filter=request.entity_filter,
            period_filter=request.period_filter,
            top_k=request.top_k
        )

        if not retrieved_facts:
            return QueryResponse(
                query=request.query,
                answer=(
                    "No verified atomic facts were found in the indexed documents matching your query. "
                    "In accordance with the system's abstention-over-wrong-answer principle, "
                    "the system abstains from generating an ungrounded or speculative answer."
                ),
                grounded_facts=[],
                related_reconciliations=[],
                confidence_note="Abstained: Zero grounded facts found in storage."
            )

        # Retrieve any relationships between these facts
        fact_ids = {f.fact_id for f in retrieved_facts}
        all_rels = storage.list_relationships(limit=200)
        related_rels = [
            r for r in all_rels
            if r.fact_a_id in fact_ids or r.fact_b_id in fact_ids
        ]

        # Prepare structured context for grounded answer synthesis
        citations: List[GroundedCitation] = []
        context_lines = []

        for i, f in enumerate(retrieved_facts, 1):
            cite = GroundedCitation(
                fact_id=f.fact_id,
                entity=f.entity,
                metric=f.metric,
                raw_value=f.raw_value,
                canonical_period=f.temporal_context.canonical_period,
                scope=f.scope,
                document_name=f.source_pointer.document_name,
                page_number=f.source_pointer.page_number,
                quoted_text=f.source_pointer.quoted_text
            )
            citations.append(cite)
            context_lines.append(
                f"[Fact {i}] Entity: {f.entity} | Metric: {f.metric} | Value: {f.raw_value} {f.unit} | "
                f"Period: {f.temporal_context.canonical_period} | Scope: {f.scope} | "
                f"Source: {f.source_pointer.document_name} (Page {f.source_pointer.page_number}) | "
                f"Quote: \"{f.source_pointer.quoted_text}\""
            )

        rel_context_lines = []
        for r in related_rels[:5]:
            rel_context_lines.append(
                f"- Relationship: {r.relation_type} ({r.explanation})"
            )

        rag_prompt = f"""You are a grounded financial knowledge assistant.
Answer the user's question USING ONLY THE STRUCTURED FACTS AND RECONCILIATIONS BELOW.

CRITICAL RULES:
1. NEVER invent, extrapolate, or estimate numbers. Only state numbers present in the facts below.
2. Cite the exact Fact number (e.g. [Fact 1]) and document page whenever stating a figure.
3. If facts corroborate each other, mention that the figure is cross-verified across multiple documents.
4. If there is a contradiction or contextual difference (e.g., consolidated vs standalone, or different fiscal years), explicitly state that difference.
5. If the facts are insufficient to answer completely, explicitly state what is missing rather than guessing.

RETRIEVED ATOMIC FACTS:
{chr(10).join(context_lines)}

CROSS-DOCUMENT RECONCILIATIONS:
{chr(10).join(rel_context_lines) if rel_context_lines else "None recorded"}

USER QUESTION: {request.query}

GROUNDED FACTUAL ANSWER:"""

        # Call LLM for natural phrasing (strictly constrained by facts)
        answer_text = None
        if self.api_key:
            headers = {"Content-Type": "application/json", "X-goog-api-key": self.api_key}
            payload = {
                "contents": [{"parts": [{"text": rag_prompt}]}],
                "generationConfig": {"temperature": 0.0}
            }
            try:
                res = requests.post(self.endpoint, headers=headers, json=payload, timeout=20)
                if res.status_code == 200:
                    cand = res.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")
                    if cand:
                        answer_text = cand.strip()
            except Exception:
                pass

        # Fallback deterministic answer synthesis if LLM is unavailable
        if not answer_text:
            answer_text = self._deterministic_synthesize(request.query, retrieved_facts, related_rels)

        return QueryResponse(
            query=request.query,
            answer=answer_text,
            grounded_facts=citations,
            related_reconciliations=[
                {
                    "relationship_id": r.relationship_id,
                    "relation_type": r.relation_type,
                    "explanation": r.explanation,
                    "difference_field": r.difference_field
                } for r in related_rels[:5]
            ],
            confidence_note="Answer synthesized exclusively around verified atomic facts with explicit document citations."
        )

    def _deterministic_synthesize(
        self,
        query: str,
        facts: List[AtomicFact],
        rels: List[Any]
    ) -> str:
        """Deterministic fallback synthesizer when LLM is offline"""
        lines = [f"Based on the verified atomic facts in the knowledge layer:"]
        for i, f in enumerate(facts[:5], 1):
            lines.append(
                f"- [Fact {i}] {f.entity} reports {f.metric} as {f.raw_value} {f.unit} "
                f"for {f.temporal_context.canonical_period} ({f.scope}) in {f.source_pointer.document_name} (Page {f.source_pointer.page_number})."
            )

        if rels:
            lines.append("\nCross-document reconciliations:")
            for r in rels[:3]:
                lines.append(f"- {r.relation_type}: {r.explanation}")

        return "\n".join(lines)


# Singleton instance
retriever = FactRetriever()
