from __future__ import annotations
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

# Temporal Context Representation
PeriodType = Literal[
    "point",
    "quarter",
    "fiscal_year",
    "calendar_year",
    "range",
    "multi_year",
    "unknown"
]

ScopeType = Literal[
    "consolidated",
    "standalone",
    "segment",
    "geography",
    "institutional",
    "general"
]

RelationshipType = Literal[
    "CORROBORATED",
    "CONTRADICTED",
    "CONTEXTUAL_DIFFERENCE",
    "UNRESOLVED"
]

FailureType = Literal[
    "MISSING_UNIT",
    "AMBIGUOUS_METRIC",
    "AMBIGUOUS_SCOPE",
    "UNRESOLVED_TEMPORAL",
    "UNGROUNDED_CLAIM",
    "TABLE_HEADER_MISMATCH",
    "EXTRACTION_ABSTAINED"
]


class TemporalContext(BaseModel):
    period_type: PeriodType = "unknown"
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None    # YYYY-MM-DD
    canonical_period: str             # e.g., "FY2024", "Q4_FY2024", "2024-03-31"
    raw_period_text: Optional[str] = None


class SourcePointer(BaseModel):
    document_id: str
    document_name: str
    page_number: int
    block_id: str
    quoted_text: str
    bbox: Optional[List[float]] = None  # [x0, y0, x1, y1] normalized or points


class AtomicFact(BaseModel):
    fact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    page_number: int
    block_id: str
    entity: str
    metric: str
    raw_value: str
    unit: str
    normalized_value: Optional[float] = None
    canonical_unit: Optional[str] = None
    temporal_context: TemporalContext
    scope: ScopeType = "general"
    scope_detail: Optional[str] = None
    source_pointer: SourcePointer
    confidence_score: float = 1.0
    confidence_breakdown: Optional[Dict[str, float]] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class LLMExtractedClaim(BaseModel):
    """Raw claim model output by LLM before server-side resolution and normalization"""
    block_id: str
    entity: str
    metric: str
    raw_value: str
    unit: str
    temporal_expression: str
    scope: ScopeType = "general"
    scope_detail: Optional[str] = None
    quoted_text: str
    is_ambiguous: bool = False
    ambiguity_reason: Optional[str] = None


class LLMExtractionBatch(BaseModel):
    claims: List[LLMExtractedClaim] = Field(default_factory=list)
    abstentions: List[Dict[str, Any]] = Field(default_factory=list)


class FactRelationship(BaseModel):
    relationship_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    fact_a_id: str
    fact_b_id: str
    relation_type: RelationshipType
    confidence: float = 1.0
    explanation: str
    difference_field: Optional[str] = None  # "scope", "period", "value", "metric", etc.
    fact_a: Optional[AtomicFact] = None
    fact_b: Optional[AtomicFact] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ExtractionFailure(BaseModel):
    failure_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    document_name: str
    page_number: int
    block_id: str
    failure_type: FailureType
    reason: str
    raw_content: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    file_path: str
    sha256: str
    page_count: int
    session_id: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str = "indexed"
    fact_count: int = 0
    relationship_count: int = 0
    failure_count: int = 0


class BlockRecord(BaseModel):
    block_id: str
    document_id: str
    document_name: str
    page_number: int
    block_type: Literal["text", "table", "header", "callout", "footnote"]
    text: str
    table_data: Optional[List[List[str]]] = None
    bbox: Optional[List[float]] = None


class PageRecord(BaseModel):
    document_id: str
    document_name: str
    page_number: int
    raw_text: str
    block_count: int
    image_path: str


class FactFilter(BaseModel):
    entity: Optional[str] = None
    metric: Optional[str] = None
    document_id: Optional[str] = None
    canonical_period: Optional[str] = None
    scope: Optional[str] = None
    min_confidence: Optional[float] = None
    search_query: Optional[str] = None
    limit: int = 100
    offset: int = 0


class QueryRequest(BaseModel):
    query: str
    entity_filter: Optional[str] = None
    period_filter: Optional[str] = None
    top_k: int = 10


class GroundedCitation(BaseModel):
    fact_id: str
    entity: str
    metric: str
    raw_value: str
    canonical_period: str
    scope: str
    document_name: str
    page_number: int
    quoted_text: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    grounded_facts: List[GroundedCitation]
    related_reconciliations: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_note: str = "Answer synthesized exclusively from verified atomic facts."
