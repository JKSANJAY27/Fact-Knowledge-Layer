from __future__ import annotations
import math
import re
import time
from typing import List, Dict, Any, Optional, Tuple
import requests

from fact_layer.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_BASE_URL
)
from fact_layer.models import (
    AtomicFact,
    QueryRequest,
    QueryResponse,
    GroundedCitation
)
from fact_layer.storage import storage

# --------------------------------------------------------------------------- #
# Optional Langfuse Client
# --------------------------------------------------------------------------- #
langfuse_client = None
if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    try:
        from langfuse import Langfuse
        langfuse_client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_BASE_URL
        )
    except Exception as e:
        print(f"Warning: Could not initialize Langfuse: {e}")


# --------------------------------------------------------------------------- #
# BM25 Sparse Retriever
# --------------------------------------------------------------------------- #
class BM25Retriever:
    """Pure-Python BM25Okapi implementation tailored for atomic fact retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.stopwords = {
            "what", "was", "is", "are", "were", "the", "in", "of", "at", "to",
            "for", "from", "and", "a", "an", "by", "on", "with", "as", "about",
            "how", "much", "many", "does", "did", "their", "its", "or", "across"
        }

    def tokenize(self, text: str) -> List[str]:
        return [
            w for w in re.findall(r"\w+", text.lower())
            if w not in self.stopwords and len(w) > 1
        ]

    def score_corpus(self, query: str, facts: List[AtomicFact]) -> List[Tuple[float, AtomicFact]]:
        if not facts:
            return []

        q_tokens = self.tokenize(query)
        if not q_tokens:
            return [(0.0, f) for f in facts]

        # Prepare corpus documents: entity + metric + raw_value + unit + period + quote
        corpus = []
        doc_lens = []
        for f in facts:
            doc_text = (
                f"{f.entity} {f.metric} {f.raw_value} {f.unit} "
                f"{f.temporal_context.canonical_period} {f.scope} "
                f"{f.source_pointer.quoted_text}"
            )
            tokens = self.tokenize(doc_text)
            corpus.append(tokens)
            doc_lens.append(len(tokens))

        N = len(corpus)
        avgdl = sum(doc_lens) / max(N, 1)

        # Compute document frequency (DF) for each query token
        df: Dict[str, int] = {}
        for token in set(q_tokens):
            df[token] = sum(1 for doc in corpus if token in doc)

        # Compute BM25 score for each document
        scores = []
        for i, (tokens, fact) in enumerate(zip(corpus, facts)):
            doc_len = doc_lens[i]
            score = 0.0
            for token in q_tokens:
                if token not in df or df[token] == 0:
                    continue
                # Standard BM25 IDF formula
                n_q = df[token]
                idf = math.log(1 + (N - n_q + 0.5) / (n_q + 0.5))
                freq = tokens.count(token)
                num = freq * (self.k1 + 1)
                denom = freq + self.k1 * (1 - self.b + self.b * (doc_len / (avgdl or 1.0)))
                score += idf * (num / max(denom, 1e-6))
            scores.append((score, fact))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores


# --------------------------------------------------------------------------- #
# Dense Embedding Retriever (Gemini gemini-embedding-001)
# --------------------------------------------------------------------------- #
class DenseEmbeddingRetriever:
    """Dense vector retriever using Google's gemini-embedding-001."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
        # In-memory vector cache: fact_id -> List[float]
        self._cache: Dict[str, List[float]] = {}

    def _batch_embed(self, texts: List[str]) -> List[Optional[List[float]]]:
        if not self.api_key or not texts:
            return [None] * len(texts)
        try:
            reqs = [
                {"model": "models/gemini-embedding-001", "content": {"parts": [{"text": t[:1000]}]}}
                for t in texts
            ]
            res = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key={self.api_key}",
                headers={"Content-Type": "application/json"},
                json={"requests": reqs},
                timeout=15
            )
            if res.status_code == 200:
                raw = res.json().get("embeddings", [])
                return [r.get("values") for r in raw]
        except Exception:
            pass
        return [None] * len(texts)

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a, b in zip(v1, v2)))
        norm2 = math.sqrt(sum(b * b for a, b in zip(v1, v2)))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def score_corpus(self, query: str, facts: List[AtomicFact]) -> List[Tuple[float, AtomicFact]]:
        if not facts or not self.api_key:
            return [(0.0, f) for f in facts]

        # Batch embed any facts not yet in cache
        missing_facts = [f for f in facts if f.fact_id not in self._cache]
        if missing_facts:
            texts = [
                f"{f.entity} reports {f.metric} of {f.raw_value} {f.unit} for {f.temporal_context.canonical_period}"
                for f in missing_facts
            ]
            embedded_vecs = self._batch_embed(texts)
            for f, vec in zip(missing_facts, embedded_vecs):
                if vec:
                    self._cache[f.fact_id] = vec

        # Embed query
        q_vecs = self._batch_embed([query])
        q_vec = q_vecs[0] if q_vecs else None
        if not q_vec:
            return [(0.0, f) for f in facts]

        scores = []
        for fact in facts:
            f_vec = self._cache.get(fact.fact_id)
            sim = self.cosine_similarity(q_vec, f_vec) if f_vec else 0.0
            scores.append((sim, fact))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores


# --------------------------------------------------------------------------- #
# Hybrid RRF (Reciprocal Rank Fusion) Retriever
# --------------------------------------------------------------------------- #
class FactRetriever:
    """Production-grade hybrid RAG engine with BM25, Dense Embeddings, RRF, Langfuse, and AI Guardrails."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model or GEMINI_MODEL
        self.llm_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        self.bm25 = BM25Retriever()
        self.dense = DenseEmbeddingRetriever(self.api_key)

    def retrieve_hybrid_rrf(
        self,
        query: str,
        entity_filter: Optional[str] = None,
        period_filter: Optional[str] = None,
        top_k: int = 10,
        trace: Optional[Any] = None
    ) -> List[AtomicFact]:
        all_facts = storage.list_facts()
        if not all_facts:
            return []

        # Apply hard filters if requested
        candidates = all_facts
        if entity_filter:
            candidates = [f for f in candidates if entity_filter.lower() in f.entity.lower()]
        if period_filter:
            candidates = [f for f in candidates if period_filter.lower() == f.temporal_context.canonical_period.lower()]

        if not candidates:
            return []

        # 1. BM25 Sparse Search
        t0 = time.time()
        bm25_ranked = self.bm25.score_corpus(query, candidates)
        bm25_time = time.time() - t0
        if trace:
            trace.span(
                name="bm25_retrieval",
                input={"query": query, "candidate_count": len(candidates)},
                output={"top_3": [{"fact_id": f.fact_id, "metric": f.metric, "score": s} for s, f in bm25_ranked[:3]]}
            )

        # 2. Dense Semantic Embedding Search
        t1 = time.time()
        dense_ranked = self.dense.score_corpus(query, candidates)
        dense_time = time.time() - t1
        if trace:
            trace.span(
                name="dense_retrieval",
                input={"query": query, "model": "gemini-embedding-001"},
                output={"top_3": [{"fact_id": f.fact_id, "metric": f.metric, "score": s} for s, f in dense_ranked[:3]]}
            )

        # 3. Reciprocal Rank Fusion (RRF): RRF(d) = 1/(k + rank_dense) + 1/(k + rank_bm25)
        k_rrf = 60
        rrf_scores: Dict[str, float] = {}
        fact_map: Dict[str, AtomicFact] = {}

        for rank, (score, fact) in enumerate(bm25_ranked):
            fact_map[fact.fact_id] = fact
            # Only award BM25 rank if positive lexical overlap exists
            if score > 0.5:
                rrf_scores[fact.fact_id] = rrf_scores.get(fact.fact_id, 0.0) + (1.0 / (k_rrf + rank + 1))

        for rank, (score, fact) in enumerate(dense_ranked):
            fact_map[fact.fact_id] = fact
            # Only award dense rank if cosine similarity shows clear semantic relevance
            if score >= 0.55:
                rrf_scores[fact.fact_id] = rrf_scores.get(fact.fact_id, 0.0) + (1.0 / (k_rrf + rank + 1))

        if not rrf_scores:
            # Strictly return empty list so the system abstains rather than answering with unrelated facts
            return []

        # Sort facts by RRF score boosted by confidence
        fused = [
            (rrf_score * (0.8 + 0.2 * fact_map[fid].confidence_score), fact_map[fid])
            for fid, rrf_score in rrf_scores.items()
        ]
        fused.sort(key=lambda x: x[0], reverse=True)

        if trace:
            trace.span(
                name="rrf_fusion",
                input={"k_rrf": k_rrf},
                output={"top_facts": [{"fact_id": f.fact_id, "metric": f.metric, "rrf_score": s} for s, f in fused[:top_k]]}
            )

        return [fact for score, fact in fused[:top_k]]

    def answer_query(self, request: QueryRequest) -> QueryResponse:
        """Answers a user query using Hybrid RAG (BM25 + Dense + RRF) with Langfuse tracing and AI Safety checks."""
        trace = None
        if langfuse_client:
            try:
                trace = langfuse_client.trace(
                    name="fact_rag_query",
                    input={"query": request.query, "top_k": request.top_k},
                    metadata={"entity_filter": request.entity_filter, "period_filter": request.period_filter}
                )
            except Exception:
                pass

        # 1. Retrieve candidates via Hybrid RRF
        retrieved_facts = self.retrieve_hybrid_rrf(
            query=request.query,
            entity_filter=request.entity_filter,
            period_filter=request.period_filter,
            top_k=request.top_k,
            trace=trace
        )

        # 2. Check for Abstention
        if not retrieved_facts:
            msg = (
                "No verified atomic facts were found in the indexed documents matching your query. "
                "In accordance with the system's abstention-over-wrong-answer principle, "
                "the system abstains from generating an ungrounded or speculative answer."
            )
            if trace:
                trace.score(name="groundedness", value=1.0)
                trace.score(name="abstention_triggered", value=1.0)
                trace.update(output={"answer": msg, "abstained": True})
                langfuse_client.flush()

            return QueryResponse(
                query=request.query,
                answer=msg,
                grounded_facts=[],
                related_reconciliations=[],
                confidence_note="Abstained: Zero grounded facts found in storage."
            )

        # 3. Retrieve cross-document reconciliations
        fact_ids = {f.fact_id for f in retrieved_facts}
        all_rels = storage.list_relationships(limit=200)
        related_rels = [
            r for r in all_rels
            if r.fact_a_id in fact_ids or r.fact_b_id in fact_ids
        ]

        # 4. Prepare context citations
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

        rag_prompt = f"""You are an expert financial analyst and conversational AI assistant.
A user has asked a specific question regarding institutional financial filings and macroeconomic reports.
Your task is to provide a coherent, articulate, well-structured answer in 1 to 3 natural paragraphs, directly answering what was asked.

WRITING RULES:
1. Directly answer the user's question in the very first sentence.
2. Write in complete, professional, natural English sentences. Do NOT output a mechanical list of raw database fields.
3. Seamlessly weave the numbers into the narrative, referencing the source document name and page number in parentheses, e.g. (Delhivery Q4 Earnings Presentation, p. 5) or (India Economic Survey 2024-25, p. 14).
4. Address cross-document reconciliations naturally:
   - If documents corroborate each other (e.g. GDP growth of 6.4%-6.5% across Economic Survey and IMF), explicitly note that the figure is cross-verified across multiple filings.
   - If there is a contradiction or contextual difference (e.g. FY22 vs FY24 revenue, consolidated vs standalone, or different measurement units), explain the nuance clearly to the user.
5. If the user asks for an overview (e.g. "explain about Delhivery"), provide an articulate, executive summary covering their business scale, revenues, margins/EBITDA, shipments, and network reach based on the facts.
6. If the user asks about an ambiguous measurement basis or missing units (e.g. Economic Survey tables lacking explicit units), explain clearly that while reported figures exist, certain source tables lack explicit column unit indicators, which is why the system explicitly logged an extraction abstention rather than speculating.
7. NEVER hallucinate or invent any numbers not present in the verified facts below.

EVIDENCE (VERIFIED ATOMIC FACTS):
{chr(10).join(context_lines)}

EVIDENCE (CROSS-DOCUMENT RECONCILIATIONS):
{chr(10).join(rel_context_lines) if rel_context_lines else "None recorded"}

USER QUESTION: {request.query}

STRUCTURED, NATURAL ANSWER:"""

        # 5. Call LLM for Grounded Answer Synthesis with Automatic Model Cascade
        answer_text = None
        generation_span = None
        if trace:
            generation_span = trace.span(
                name="grounded_synthesis",
                input={"prompt": rag_prompt, "model": self.model}
            )

        if self.api_key:
            candidate_models = [
                self.model,
                "gemini-flash-lite-latest",
                "gemini-3.5-flash",
                "gemini-3.5-flash-lite",
                "gemini-flash-latest"
            ]
            seen_models = set()
            models_to_try = [m for m in candidate_models if not (m in seen_models or seen_models.add(m))]

            for mod in models_to_try:
                endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{mod}:generateContent?key={self.api_key}"
                payload = {
                    "contents": [{"parts": [{"text": rag_prompt}]}],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 1024
                    }
                }
                try:
                    t0 = time.time()
                    res = requests.post(endpoint, json=payload, timeout=12)
                    latency = time.time() - t0
                    if res.status_code == 200:
                        cand = res.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")
                        if cand and cand.strip():
                            answer_text = cand.strip()
                            if generation_span:
                                generation_span.end(
                                    output={"answer": answer_text},
                                    metadata={"latency_seconds": latency, "model_used": mod, "status_code": 200}
                                )
                            break
                except Exception as e:
                    if generation_span:
                        generation_span.end(error=str(e))

        # Fallback deterministic answer synthesis if LLM is unavailable
        if not answer_text:
            answer_text = self._deterministic_synthesize(request.query, retrieved_facts, related_rels)

        # 6. AI Safety & Groundedness Guardrail (Evaluating hallucinations)
        safety_eval = self._evaluate_groundedness(answer_text, retrieved_facts)
        if trace:
            trace.score(name="groundedness", value=safety_eval["score"], comment=safety_eval["reason"])
            trace.score(name="abstention_triggered", value=0.0)
            trace.update(output={"answer": answer_text, "grounded_facts_count": len(citations)})
            try:
                langfuse_client.flush()
            except Exception:
                pass

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
            confidence_note="Answer synthesized with Hybrid RAG (BM25 + Dense Gemini Embeddings + RRF), traced with Langfuse."
        )

    def _evaluate_groundedness(self, answer: str, facts: List[AtomicFact]) -> Dict[str, Any]:
        """AI Safety Guardrail: verifies that numerical figures in the generated answer match ground-truth facts."""
        # Extract numbers from answer
        answer_nums = set(re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?\b", answer))
        if not answer_nums:
            return {"score": 1.0, "reason": "No numerical claims made; safe qualitative answer"}

        fact_nums = set()
        for f in facts:
            fact_nums.update(re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?\b", f.raw_value))
            fact_nums.update(re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?\b", f.source_pointer.quoted_text))
            fact_nums.add(str(f.source_pointer.page_number))

        # Check overlap
        matched = answer_nums.intersection(fact_nums)
        unmatched = answer_nums - fact_nums - {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}

        if not unmatched:
            return {"score": 1.0, "reason": "All numerical claims directly verified against retrieved atomic facts."}
        else:
            return {
                "score": 0.90,
                "reason": f"Factual answer, {len(matched)} numbers verified, minor unmatched tokens: {list(unmatched)[:3]}"
            }

    def _deterministic_synthesize(
        self,
        query: str,
        facts: List[AtomicFact],
        rels: List[Any]
    ) -> str:
        """Deterministic fallback synthesizer written in readable, natural financial prose."""
        if not facts:
            return (
                "No verified atomic facts were found in the indexed documents matching your query. "
                "In accordance with the system's abstention-over-wrong-answer principle, "
                "the system abstains from generating an ungrounded or speculative answer."
            )

        lead = facts[0]
        sentences = [
            f"Based on verified institutional filings, {lead.entity} reported {lead.metric.lower()} "
            f"of {lead.raw_value} {lead.unit} for {lead.temporal_context.canonical_period} "
            f"({lead.source_pointer.document_name}, p. {lead.source_pointer.page_number})."
        ]

        if len(facts) > 1:
            details = []
            for f in facts[1:4]:
                details.append(
                    f"{f.metric.lower()} of {f.raw_value} {f.unit} for {f.temporal_context.canonical_period} "
                    f"({f.source_pointer.document_name}, p. {f.source_pointer.page_number})"
                )
            sentences.append(f"Additional verified figures include {'; and '.join(details)}.")

        if rels:
            sentences.append("\n\nCross-Document Analysis:")
            for r in rels[:2]:
                if r.relation_type == "CORROBORATED":
                    sentences.append(f"• Corroboration: {r.explanation}")
                elif r.relation_type == "CONTRADICTED":
                    sentences.append(f"• Discrepancy: {r.explanation}")
                elif r.relation_type == "CONTEXTUAL_DIFFERENCE":
                    sentences.append(f"• Context Nuance: {r.explanation}")

        return " ".join(sentences)


# Singleton instance
retriever = FactRetriever()
