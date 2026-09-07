from __future__ import annotations
import hashlib
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
import pdfplumber

from fact_layer.config import PAGES_DIR, DATA_DIR
from fact_layer.models import DocumentRecord, PageRecord, BlockRecord
from fact_layer.storage import storage


def compute_file_sha256(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


class PDFIngestor:
    def __init__(self):
        PAGES_DIR.mkdir(parents=True, exist_ok=True)

    def ingest_pdf(
        self,
        file_path: Path,
        max_pages: Optional[int] = None,
        force_reprocess: bool = False
    ) -> Tuple[DocumentRecord, List[PageRecord], List[BlockRecord]]:
        """
        Ingests a PDF file page-by-page:
        - Checks sha256 to avoid reprocessing existing documents (incremental ingestion).
        - Extracts text blocks, layout blocks, and tables.
        - Renders and persists page images.
        - Assigns deterministic block IDs: blk_{doc_prefix}_p{page}_{idx}.
        - Never discards source pointers.
        """
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        file_hash = compute_file_sha256(file_path)
        existing_doc = storage.get_document_by_hash(file_hash)

        if existing_doc and not force_reprocess:
            pages = storage.get_pages_for_doc(existing_doc.document_id)
            blocks = []
            for p in pages:
                blocks.extend(storage.get_blocks_for_page(existing_doc.document_id, p.page_number))
            return existing_doc, pages, blocks

        doc_prefix = file_hash[:10]
        doc_id = f"doc_{doc_prefix}"
        filename = file_path.name

        doc_fitz = fitz.open(str(file_path))
        total_pages = len(doc_fitz)
        pages_to_process = min(total_pages, max_pages) if max_pages else total_pages

        document_record = DocumentRecord(
            document_id=doc_id,
            filename=filename,
            file_path=str(file_path),
            sha256=file_hash,
            page_count=pages_to_process,
            status="ingesting"
        )
        storage.add_document(document_record)

        all_page_records: List[PageRecord] = []
        all_block_records: List[BlockRecord] = []

        # Open pdfplumber for table parsing
        plumber_doc = pdfplumber.open(str(file_path))

        try:
            for page_idx in range(pages_to_process):
                page_num = page_idx + 1
                fitz_page = doc_fitz[page_idx]
                plumber_page = plumber_doc.pages[page_idx] if page_idx < len(plumber_doc.pages) else None

                # 1. Render page image
                image_rel_path = f"pages/{doc_id}_p{page_num}.png"
                image_abs_path = DATA_DIR / image_rel_path
                if not image_abs_path.exists():
                    pix = fitz_page.get_pixmap(dpi=140)
                    pix.save(str(image_abs_path))

                # 2. Extract tables via pdfplumber
                page_tables = []
                table_bboxes = []
                if plumber_page:
                    extracted_tables = plumber_page.extract_tables()
                    # Also find bounding boxes of tables to avoid duplicating text
                    for t in plumber_page.find_tables():
                        table_bboxes.append(list(t.bbox))
                    for t in extracted_tables:
                        if t and len(t) > 1:
                            # Filter empty rows/cols
                            cleaned_t = [[str(cell).strip() if cell is not None else "" for cell in row] for row in t]
                            # Only keep if has content
                            if any(any(c for c in r) for r in cleaned_t):
                                page_tables.append(cleaned_t)

                # 3. Extract text blocks via PyMuPDF
                # get_text("blocks") returns (x0, y0, x1, y1, "text", block_no, block_type)
                raw_blocks = fitz_page.get_text("blocks")
                page_raw_text = fitz_page.get_text("text")

                page_block_records: List[BlockRecord] = []
                block_seq = 1

                # Add extracted tables as high-fidelity table blocks first
                for t_idx, table_data in enumerate(page_tables):
                    t_bbox = table_bboxes[t_idx] if t_idx < len(table_bboxes) else None
                    # Format as markdown-like text
                    table_str_rows = [" | ".join(row) for row in table_data]
                    table_text = "\n".join(table_str_rows)

                    block_id = f"blk_{doc_prefix}_p{page_num}_t{block_seq}"
                    b_rec = BlockRecord(
                        block_id=block_id,
                        document_id=doc_id,
                        document_name=filename,
                        page_number=page_num,
                        block_type="table",
                        text=table_text,
                        table_data=table_data,
                        bbox=t_bbox
                    )
                    page_block_records.append(b_rec)
                    block_seq += 1

                # Add text blocks
                for rb in raw_blocks:
                    bx0, by0, bx1, by1, btext, bno, btype = rb[:7]
                    clean_btext = btext.strip()
                    if not clean_btext:
                        continue

                    # Check if block is inside a table bbox
                    in_table = False
                    for tb in table_bboxes:
                        tx0, ty0, tx1, ty1 = tb
                        if bx0 >= tx0 - 5 and by0 >= ty0 - 5 and bx1 <= tx1 + 5 and by1 <= ty1 + 5:
                            in_table = True
                            break
                    if in_table and page_tables:
                        continue

                    # Classify block type
                    block_type = "text"
                    if len(clean_btext.splitlines()) == 1 and len(clean_btext) < 80 and (clean_btext.isupper() or clean_btext.istitle()):
                        block_type = "header"
                    elif "note" in clean_btext.lower()[:20] or clean_btext.startswith("*"):
                        block_type = "footnote"

                    block_id = f"blk_{doc_prefix}_p{page_num}_b{block_seq}"
                    b_rec = BlockRecord(
                        block_id=block_id,
                        document_id=doc_id,
                        document_name=filename,
                        page_number=page_num,
                        block_type=block_type,
                        text=clean_btext,
                        table_data=None,
                        bbox=[bx0, by0, bx1, by1]
                    )
                    page_block_records.append(b_rec)
                    block_seq += 1

                # 4. Create PageRecord
                p_rec = PageRecord(
                    document_id=doc_id,
                    document_name=filename,
                    page_number=page_num,
                    raw_text=page_raw_text,
                    block_count=len(page_block_records),
                    image_path=image_rel_path
                )
                storage.add_page(p_rec)
                storage.add_blocks(page_block_records)

                all_page_records.append(p_rec)
                all_block_records.extend(page_block_records)

        finally:
            doc_fitz.close()
            plumber_doc.close()

        # Update document status
        document_record.status = "indexed"
        storage.add_document(document_record)

        return document_record, all_page_records, all_block_records


# Singleton instance
ingestor = PDFIngestor()
