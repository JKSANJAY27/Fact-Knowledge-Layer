"""
Deduplicate facts in the database.
Facts are considered duplicates if they have the same:
  (block_id, entity, metric, raw_value, canonical_period, scope)

Keeps only the highest-confidence fact per unique combination.
Also deduplicates relationships.
"""
import sys
from pathlib import Path
import sqlite3

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from fact_layer.config import DB_PATH

conn = sqlite3.connect(str(DB_PATH))
conn.row_factory = sqlite3.Row

print("=== Before dedup ===")
count_before = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
print(f"  Facts: {count_before}")
rel_before = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
print(f"  Relationships: {rel_before}")

# --- Deduplicate facts ---
# For each (block_id, entity, metric, raw_value, canonical_period, scope) group,
# keep only the row with the highest confidence_score (and lowest ROWID for ties)
print("\nDeduplicating facts...")
conn.execute("""
DELETE FROM facts
WHERE fact_id NOT IN (
    SELECT fact_id FROM (
        SELECT fact_id,
               ROW_NUMBER() OVER (
                   PARTITION BY block_id, entity, metric, raw_value, canonical_period, scope
                   ORDER BY confidence_score DESC, rowid ASC
               ) AS rn
        FROM facts
    ) ranked
    WHERE rn = 1
)
""")
conn.commit()

count_after = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
print(f"  Facts after dedup: {count_after} (removed {count_before - count_after} duplicates)")

# --- Deduplicate relationships (keep distinct fact_a_id, fact_b_id, relation_type combinations) ---
print("\nDeduplicating relationships...")
conn.execute("""
DELETE FROM relationships
WHERE relationship_id NOT IN (
    SELECT relationship_id FROM (
        SELECT relationship_id,
               ROW_NUMBER() OVER (
                   PARTITION BY 
                       CASE WHEN fact_a_id < fact_b_id THEN fact_a_id ELSE fact_b_id END,
                       CASE WHEN fact_a_id < fact_b_id THEN fact_b_id ELSE fact_a_id END,
                       relation_type
                   ORDER BY rowid ASC
               ) AS rn
        FROM relationships
    ) ranked
    WHERE rn = 1
)
""")
conn.commit()

# Also remove relationships where either fact was deleted
conn.execute("""
DELETE FROM relationships
WHERE fact_a_id NOT IN (SELECT fact_id FROM facts)
   OR fact_b_id NOT IN (SELECT fact_id FROM facts)
""")
conn.commit()

# Remove self-referential relationships (same document)
conn.execute("""
DELETE FROM relationships
WHERE fact_a_id IN (
    SELECT f.fact_id FROM facts f WHERE f.document_id = (
        SELECT f2.document_id FROM facts f2 WHERE f2.fact_id = relationships.fact_b_id
    )
)
AND fact_b_id IN (
    SELECT f.fact_id FROM facts f WHERE f.document_id = (
        SELECT f2.document_id FROM facts f2 WHERE f2.fact_id = relationships.fact_a_id
    )
)
AND (
    SELECT document_id FROM facts WHERE fact_id = relationships.fact_a_id
) = (
    SELECT document_id FROM facts WHERE fact_id = relationships.fact_b_id
)
""")
conn.commit()

rel_after = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
print(f"  Relationships after dedup: {rel_after} (removed {rel_before - rel_after})")

print("\n=== After dedup ===")
print(f"  Facts: {conn.execute('SELECT COUNT(*) FROM facts').fetchone()[0]}")
print(f"  Relationships: {conn.execute('SELECT COUNT(*) FROM relationships').fetchone()[0]}")

# Show relationship breakdown
for row in conn.execute("SELECT relation_type, COUNT(*) as cnt FROM relationships GROUP BY relation_type").fetchall():
    print(f"    {row['relation_type']}: {row['cnt']}")

print("\nFinal facts:")
for row in conn.execute("""
    SELECT DISTINCT entity, metric, raw_value, unit, canonical_period, scope, 
           d.filename, page_number
    FROM facts f
    LEFT JOIN documents d ON f.document_id = d.document_id
    ORDER BY entity, metric
""").fetchall():
    print(f"  {row['entity']} | {row['metric']} = {row['raw_value']} {row['unit']} | {row['canonical_period']} | {row['filename'][:40]} p{row['page_number']}")

conn.close()
print("\nDone.")
