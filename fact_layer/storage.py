from __future__ import annotations
import sqlite3
import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from fact_layer.config import DB_PATH, DATA_DIR
from fact_layer.models import (
    DocumentRecord,
    PageRecord,
    BlockRecord,
    AtomicFact,
    TemporalContext,
    SourcePointer,
    FactRelationship,
    ExtractionFailure,
    FactFilter
)


class FactStorage:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self._init_db()
        self._embedder = None

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Documents table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                sha256 TEXT UNIQUE NOT NULL,
                page_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL
            );
            """)

            # Pages table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS pages (
                document_id TEXT NOT NULL,
                document_name TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                raw_text TEXT NOT NULL,
                block_count INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                PRIMARY KEY (document_id, page_number),
                FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
            );
            """)

            # Blocks table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS blocks (
                block_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                document_name TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                block_type TEXT NOT NULL,
                text TEXT NOT NULL,
                table_data_json TEXT,
                bbox_json TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
            );
            """)

            # Facts table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                block_id TEXT NOT NULL,
                entity TEXT NOT NULL,
                metric TEXT NOT NULL,
                raw_value TEXT NOT NULL,
                unit TEXT NOT NULL,
                normalized_value REAL,
                canonical_unit TEXT,
                period_type TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT,
                canonical_period TEXT NOT NULL,
                raw_period_text TEXT,
                scope TEXT NOT NULL,
                scope_detail TEXT,
                quoted_text TEXT NOT NULL,
                bbox_json TEXT,
                confidence_score REAL NOT NULL,
                confidence_breakdown_json TEXT,
                created_at TEXT NOT NULL,
                embedding_json TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE,
                FOREIGN KEY (block_id) REFERENCES blocks(block_id) ON DELETE CASCADE
            );
            """)

            # Relationships table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                relationship_id TEXT PRIMARY KEY,
                fact_a_id TEXT NOT NULL,
                fact_b_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                explanation TEXT NOT NULL,
                difference_field TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (fact_a_id) REFERENCES facts(fact_id) ON DELETE CASCADE,
                FOREIGN KEY (fact_b_id) REFERENCES facts(fact_id) ON DELETE CASCADE
            );
            """)

            # Extraction failures / abstentions table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS failures (
                failure_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                document_name TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                block_id TEXT NOT NULL,
                failure_type TEXT NOT NULL,
                reason TEXT NOT NULL,
                raw_content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
            );
            """)

            # Indexes for ultra-fast query & filtering
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_entity ON facts(entity);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_metric ON facts(metric);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_period ON facts(canonical_period);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_scope ON facts(scope);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_doc ON facts(document_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_fact_a ON relationships(fact_a_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_fact_b ON relationships(fact_b_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(relation_type);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fail_type ON failures(failure_type);")

            conn.commit()

    # --- Document Methods ---
    def get_document_by_hash(self, sha256: str) -> Optional[DocumentRecord]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
            if row:
                return self._row_to_document(row, conn)
            return None

    def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,)).fetchone()
            if row:
                return self._row_to_document(row, conn)
            return None

    def list_documents(self) -> List[DocumentRecord]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
            return [self._row_to_document(row, conn) for row in rows]

    def add_document(self, doc: DocumentRecord):
        with self._get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO documents (document_id, filename, file_path, sha256, page_count, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (doc.document_id, doc.filename, doc.file_path, doc.sha256, doc.page_count, doc.created_at, doc.status))
            conn.commit()

    def _row_to_document(self, row: sqlite3.Row, conn: sqlite3.Connection) -> DocumentRecord:
        doc_id = row["document_id"]
        fact_count = conn.execute("SELECT COUNT(*) FROM facts WHERE document_id = ?", (doc_id,)).fetchone()[0]
        rel_count = conn.execute("""
            SELECT COUNT(DISTINCT r.relationship_id) FROM relationships r
            JOIN facts f ON r.fact_a_id = f.fact_id OR r.fact_b_id = f.fact_id
            WHERE f.document_id = ?
        """, (doc_id,)).fetchone()[0]
        failure_count = conn.execute("SELECT COUNT(*) FROM failures WHERE document_id = ?", (doc_id,)).fetchone()[0]

        return DocumentRecord(
            document_id=row["document_id"],
            filename=row["filename"],
            file_path=row["file_path"],
            sha256=row["sha256"],
            page_count=row["page_count"],
            created_at=row["created_at"],
            status=row["status"],
            fact_count=fact_count,
            relationship_count=rel_count,
            failure_count=failure_count
        )

    # --- Page Methods ---
    def add_page(self, page: PageRecord):
        with self._get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO pages (document_id, document_name, page_number, raw_text, block_count, image_path)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (page.document_id, page.document_name, page.page_number, page.raw_text, page.block_count, page.image_path))
            conn.commit()

    def get_page(self, document_id: str, page_number: int) -> Optional[PageRecord]:
        with self._get_connection() as conn:
            row = conn.execute("""
            SELECT * FROM pages WHERE document_id = ? AND page_number = ?
            """, (document_id, page_number)).fetchone()
            if row:
                return PageRecord(
                    document_id=row["document_id"],
                    document_name=row["document_name"],
                    page_number=row["page_number"],
                    raw_text=row["raw_text"],
                    block_count=row["block_count"],
                    image_path=row["image_path"]
                )
            return None

    def get_pages_for_doc(self, document_id: str) -> List[PageRecord]:
        with self._get_connection() as conn:
            rows = conn.execute("""
            SELECT * FROM pages WHERE document_id = ? ORDER BY page_number ASC
            """, (document_id,)).fetchall()
            return [
                PageRecord(
                    document_id=r["document_id"],
                    document_name=r["document_name"],
                    page_number=r["page_number"],
                    raw_text=r["raw_text"],
                    block_count=r["block_count"],
                    image_path=r["image_path"]
                ) for r in rows
            ]

    # --- Block Methods ---
    def add_blocks(self, blocks: List[BlockRecord]):
        if not blocks:
            return
        with self._get_connection() as conn:
            conn.executemany("""
            INSERT OR REPLACE INTO blocks (block_id, document_id, document_name, page_number, block_type, text, table_data_json, bbox_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    b.block_id,
                    b.document_id,
                    b.document_name,
                    b.page_number,
                    b.block_type,
                    b.text,
                    json.dumps(b.table_data) if b.table_data else None,
                    json.dumps(b.bbox) if b.bbox else None
                ) for b in blocks
            ])
            conn.commit()

    def get_block(self, block_id: str) -> Optional[BlockRecord]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM blocks WHERE block_id = ?", (block_id,)).fetchone()
            if row:
                return self._row_to_block(row)
            return None

    def get_blocks_for_page(self, document_id: str, page_number: int) -> List[BlockRecord]:
        with self._get_connection() as conn:
            rows = conn.execute("""
            SELECT * FROM blocks WHERE document_id = ? AND page_number = ?
            ORDER BY block_id ASC
            """, (document_id, page_number)).fetchall()
            return [self._row_to_block(r) for r in rows]

    def _row_to_block(self, row: sqlite3.Row) -> BlockRecord:
        return BlockRecord(
            block_id=row["block_id"],
            document_id=row["document_id"],
            document_name=row["document_name"],
            page_number=row["page_number"],
            block_type=row["block_type"],
            text=row["text"],
            table_data=json.loads(row["table_data_json"]) if row["table_data_json"] else None,
            bbox=json.loads(row["bbox_json"]) if row["bbox_json"] else None
        )

    # --- Fact Methods ---
    def add_facts(self, facts: List[AtomicFact]):
        """Inserts facts with content-based deduplication.
        A fact is considered a duplicate if (block_id, entity, metric, raw_value, canonical_period, scope)
        already exists — this prevents double-insertion across multiple extraction runs.
        """
        if not facts:
            return
        with self._get_connection() as conn:
            for f in facts:
                # Check if an equivalent fact already exists by content signature
                existing = conn.execute("""
                SELECT fact_id FROM facts
                WHERE block_id = ? AND entity = ? AND metric = ? AND raw_value = ?
                  AND canonical_period = ? AND scope = ?
                LIMIT 1
                """, (
                    f.block_id, f.entity, f.metric, f.raw_value,
                    f.temporal_context.canonical_period, f.scope
                )).fetchone()

                if existing:
                    # Update confidence if the new one is higher, but don't create a duplicate
                    conn.execute("""
                    UPDATE facts SET confidence_score = MAX(confidence_score, ?)
                    WHERE fact_id = ?
                    """, (f.confidence_score, existing["fact_id"]))
                    continue

                conn.execute("""
                INSERT INTO facts (
                    fact_id, document_id, page_number, block_id, entity, metric,
                    raw_value, unit, normalized_value, canonical_unit,
                    period_type, start_date, end_date, canonical_period, raw_period_text,
                    scope, scope_detail, quoted_text, bbox_json,
                    confidence_score, confidence_breakdown_json, created_at, embedding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    f.fact_id,
                    f.document_id,
                    f.page_number,
                    f.block_id,
                    f.entity,
                    f.metric,
                    f.raw_value,
                    f.unit,
                    f.normalized_value,
                    f.canonical_unit,
                    f.temporal_context.period_type,
                    f.temporal_context.start_date,
                    f.temporal_context.end_date,
                    f.temporal_context.canonical_period,
                    f.temporal_context.raw_period_text,
                    f.scope,
                    f.scope_detail,
                    f.source_pointer.quoted_text,
                    json.dumps(f.source_pointer.bbox) if f.source_pointer.bbox else None,
                    f.confidence_score,
                    json.dumps(f.confidence_breakdown) if f.confidence_breakdown else None,
                    f.created_at,
                    None
                ))
            conn.commit()


    def get_fact(self, fact_id: str) -> Optional[AtomicFact]:
        with self._get_connection() as conn:
            row = conn.execute("""
            SELECT f.*, d.filename as document_name
            FROM facts f
            LEFT JOIN documents d ON f.document_id = d.document_id
            WHERE f.fact_id = ?
            """, (fact_id,)).fetchone()
            if row:
                return self._row_to_fact(row)
            return None

    def list_facts(self, filter_params: Optional[FactFilter] = None) -> List[AtomicFact]:
        query = """
        SELECT f.*, d.filename as document_name
        FROM facts f
        LEFT JOIN documents d ON f.document_id = d.document_id
        WHERE 1=1
        """
        params = []

        if filter_params:
            if filter_params.entity:
                query += " AND LOWER(f.entity) LIKE LOWER(?)"
                params.append(f"%{filter_params.entity}%")
            if filter_params.metric:
                query += " AND LOWER(f.metric) LIKE LOWER(?)"
                params.append(f"%{filter_params.metric}%")
            if filter_params.document_id:
                query += " AND f.document_id = ?"
                params.append(filter_params.document_id)
            if filter_params.canonical_period:
                query += " AND LOWER(f.canonical_period) = LOWER(?)"
                params.append(filter_params.canonical_period)
            if filter_params.scope:
                query += " AND LOWER(f.scope) = LOWER(?)"
                params.append(filter_params.scope)
            if filter_params.min_confidence is not None:
                query += " AND f.confidence_score >= ?"
                params.append(filter_params.min_confidence)
            if filter_params.search_query:
                query += """ AND (
                    LOWER(f.entity) LIKE LOWER(?) OR
                    LOWER(f.metric) LIKE LOWER(?) OR
                    LOWER(f.quoted_text) LIKE LOWER(?) OR
                    LOWER(f.canonical_period) LIKE LOWER(?)
                )"""
                term = f"%{filter_params.search_query}%"
                params.extend([term, term, term, term])

            query += " ORDER BY f.confidence_score DESC, f.page_number ASC"
            query += f" LIMIT {filter_params.limit} OFFSET {filter_params.offset}"
        else:
            query += " ORDER BY f.confidence_score DESC, f.page_number ASC LIMIT 200"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_fact(r) for r in rows]

    def count_facts(self) -> int:
        with self._get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    def _row_to_fact(self, row: sqlite3.Row) -> AtomicFact:
        doc_name = row["document_name"] if "document_name" in row.keys() and row["document_name"] else "Unknown Document"
        bbox = json.loads(row["bbox_json"]) if row["bbox_json"] else None
        confidence_breakdown = json.loads(row["confidence_breakdown_json"]) if row["confidence_breakdown_json"] else None

        temporal = TemporalContext(
            period_type=row["period_type"],
            start_date=row["start_date"],
            end_date=row["end_date"],
            canonical_period=row["canonical_period"],
            raw_period_text=row["raw_period_text"]
        )

        source = SourcePointer(
            document_id=row["document_id"],
            document_name=doc_name,
            page_number=row["page_number"],
            block_id=row["block_id"],
            quoted_text=row["quoted_text"],
            bbox=bbox
        )

        return AtomicFact(
            fact_id=row["fact_id"],
            document_id=row["document_id"],
            page_number=row["page_number"],
            block_id=row["block_id"],
            entity=row["entity"],
            metric=row["metric"],
            raw_value=row["raw_value"],
            unit=row["unit"],
            normalized_value=row["normalized_value"],
            canonical_unit=row["canonical_unit"],
            temporal_context=temporal,
            scope=row["scope"],
            scope_detail=row["scope_detail"],
            source_pointer=source,
            confidence_score=row["confidence_score"],
            confidence_breakdown=confidence_breakdown,
            created_at=row["created_at"]
        )

    # --- Relationship Methods ---
    def add_relationships(self, relationships: List[FactRelationship]):
        if not relationships:
            return
        with self._get_connection() as conn:
            conn.executemany("""
            INSERT OR REPLACE INTO relationships (
                relationship_id, fact_a_id, fact_b_id, relation_type, confidence,
                explanation, difference_field, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    r.relationship_id,
                    r.fact_a_id,
                    r.fact_b_id,
                    r.relation_type,
                    r.confidence,
                    r.explanation,
                    r.difference_field,
                    r.created_at
                ) for r in relationships
            ])
            conn.commit()

    def list_relationships(
        self,
        relation_type: Optional[str] = None,
        limit: int = 150,
        offset: int = 0
    ) -> List[FactRelationship]:
        query = """
        SELECT r.*
        FROM relationships r
        WHERE 1=1
        """
        params = []
        if relation_type:
            query += " AND r.relation_type = ?"
            params.append(relation_type.upper())

        query += f" ORDER BY r.confidence DESC, r.created_at DESC LIMIT {limit} OFFSET {offset}"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for r in rows:
                rel = FactRelationship(
                    relationship_id=r["relationship_id"],
                    fact_a_id=r["fact_a_id"],
                    fact_b_id=r["fact_b_id"],
                    relation_type=r["relation_type"],
                    confidence=r["confidence"],
                    explanation=r["explanation"],
                    difference_field=r["difference_field"],
                    created_at=r["created_at"],
                    fact_a=self.get_fact(r["fact_a_id"]),
                    fact_b=self.get_fact(r["fact_b_id"])
                )
                results.append(rel)
            return results

    # --- Failure Methods ---
    def add_failures(self, failures: List[ExtractionFailure]):
        if not failures:
            return
        with self._get_connection() as conn:
            conn.executemany("""
            INSERT OR REPLACE INTO failures (
                failure_id, document_id, document_name, page_number, block_id,
                failure_type, reason, raw_content, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    f.failure_id,
                    f.document_id,
                    f.document_name,
                    f.page_number,
                    f.block_id,
                    f.failure_type,
                    f.reason,
                    f.raw_content,
                    f.created_at
                ) for f in failures
            ])
            conn.commit()

    def list_failures(self, failure_type: Optional[str] = None, limit: int = 100) -> List[ExtractionFailure]:
        query = "SELECT * FROM failures WHERE 1=1"
        params = []
        if failure_type:
            query += " AND failure_type = ?"
            params.append(failure_type.upper())
        query += f" ORDER BY created_at DESC LIMIT {limit}"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                ExtractionFailure(
                    failure_id=r["failure_id"],
                    document_id=r["document_id"],
                    document_name=r["document_name"],
                    page_number=r["page_number"],
                    block_id=r["block_id"],
                    failure_type=r["failure_type"],
                    reason=r["reason"],
                    raw_content=r["raw_content"],
                    created_at=r["created_at"]
                ) for r in rows
            ]

    # --- Statistics ---
    def get_stats(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            total_pages = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
            total_blocks = conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
            total_facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

            rel_counts = {}
            for row in conn.execute("SELECT relation_type, COUNT(*) FROM relationships GROUP BY relation_type"):
                rel_counts[row[0]] = row[1]

            fail_counts = {}
            for row in conn.execute("SELECT failure_type, COUNT(*) FROM failures GROUP BY failure_type"):
                fail_counts[row[0]] = row[1]

            return {
                "total_documents": total_docs,
                "total_pages": total_pages,
                "total_blocks": total_blocks,
                "total_facts": total_facts,
                "relationships": rel_counts,
                "failures": fail_counts,
                "total_relationships": sum(rel_counts.values()),
                "total_failures": sum(fail_counts.values())
            }


# Singleton instance
storage = FactStorage()
