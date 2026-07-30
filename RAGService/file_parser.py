"""
file_parser.py — Universal file-to-text extractor.
Supported: .pptx .docx .pdf .xlsx .xls .csv .txt .md
"""
from __future__ import annotations
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

SUPPORTED_EXTENSIONS = {
    ".pptx", ".ppt", ".docx", ".doc",
    ".pdf", ".xlsx", ".xls", ".csv", ".txt", ".md",
}


def parse_file(
    file_bytes: bytes,
    filename: str,
    source: str = "",
    extra_metadata: dict | None = None,
) -> List[dict]:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported type '{ext}'")
    parser = {
        ".pptx": _pptx, ".ppt": _pptx,
        ".docx": _docx, ".doc": _docx,
        ".pdf":  _pdf,
        ".xlsx": _excel, ".xls": _excel,
        ".csv":  _csv,
        ".txt":  _text, ".md": _text,
    }[ext]
    items = parser(file_bytes, filename)
    base_meta = {
        "filename": filename,
        "source":   source,
        "date_indexed": datetime.now(timezone.utc).isoformat(),
        **(extra_metadata or {}),
    }
    return _chunk(items, base_meta)


def extract_full_text(file_bytes: bytes, filename: str) -> str:
    chunks = parse_file(file_bytes, filename)
    return "\n".join(c["text"] for c in chunks)


# ── Per-format parsers ─────────────────────────────────────────────────

def _pptx(fb, fn) -> List[dict]:
    from pptx import Presentation
    prs = Presentation(io.BytesIO(fb))
    items = []
    for i, slide in enumerate(prs.slides, 1):
        title = slide.shapes.title.text_frame.text.strip() if slide.shapes.title and slide.shapes.title.has_text_frame else ""
        body  = "\n".join(
            s.text_frame.text.strip() for s in slide.shapes
            if s.has_text_frame and s != slide.shapes.title
        )
        notes = ""
        try:
            notes = slide.notes_slide.notes_text_frame.text.strip()
        except Exception:
            pass
        text = "\n".join(filter(None, [title, body, notes]))
        if text:
            items.append({"text": text, "loc": {"slide": i, "title": title}})
    return items


def _docx(fb, fn) -> List[dict]:
    from docx import Document
    doc = Document(io.BytesIO(fb))
    items = [{"text": p.text.strip(), "loc": {"para": i}} for i, p in enumerate(doc.paragraphs, 1) if p.text.strip()]
    for ti, t in enumerate(doc.tables, 1):
        rows = [" | ".join(c.text.strip() for c in r.cells if c.text.strip()) for r in t.rows]
        txt = "\n".join(r for r in rows if r)
        if txt:
            items.append({"text": f"[Table {ti}]\n{txt}", "loc": {"table": ti}})
    return items


def _pdf(fb, fn) -> List[dict]:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(fb))
    return [{"text": (p.extract_text() or "").strip(), "loc": {"page": i}} for i, p in enumerate(reader.pages, 1) if (p.extract_text() or "").strip()]


def _excel(fb, fn) -> List[dict]:
    import pandas as pd
    sheets = pd.read_excel(io.BytesIO(fb), sheet_name=None, dtype=str)
    items = []
    for name, df in sheets.items():
        df = df.fillna("")
        header = " | ".join(str(c) for c in df.columns)
        rows = [" | ".join(f"{col}: {val}" for col, val in row.items() if val) for _, row in df.iterrows()]
        txt = f"Sheet: {name}\nColumns: {header}\n" + "\n".join(r for r in rows if r)
        if txt.strip():
            items.append({"text": txt, "loc": {"sheet": name}})
    return items


def _csv(fb, fn) -> List[dict]:
    import pandas as pd
    df = pd.read_csv(io.BytesIO(fb), dtype=str).fillna("")
    header = " | ".join(str(c) for c in df.columns)
    rows = [" | ".join(f"{col}: {val}" for col, val in row.items() if val) for _, row in df.iterrows()]
    txt = f"Columns: {header}\n" + "\n".join(r for r in rows if r)
    return [{"text": txt, "loc": {"rows": len(df)}}] if txt.strip() else []


def _text(fb, fn) -> List[dict]:
    txt = fb.decode("utf-8", errors="replace").strip()
    return [{"text": txt, "loc": {}}] if txt else []


def _chunk(items: List[dict], base_meta: dict) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    result = []
    for item in items:
        for idx, chunk_text in enumerate(splitter.split_text(item["text"])):
            result.append({
                "text": chunk_text,
                "metadata": {**base_meta, **item["loc"], "chunk_idx": idx},
            })
    return result
