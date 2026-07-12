"""
src/parsing/resume_parser.py
------------------------------
Phase 3: extract plain text from an uploaded resume (PDF, DOCX, or raw .txt).

Used by the Streamlit app — the user uploads a file, this module returns
the raw text string that then passes through clean_text() before the ML pipeline.

Supported formats: .pdf, .docx, .txt
"""

import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract all text from a PDF given its raw bytes.
    Uses pdfplumber (handles multi-column layouts better than pypdf).
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber is required for PDF parsing. Run: pip install pdfplumber")

    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    text = "\n".join(text_parts)
    logger.info("PDF extracted: %d characters across %d pages", len(text), len(text_parts))
    return text


def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extract all paragraph text from a DOCX given its raw bytes.
    Uses python-docx.
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx is required for DOCX parsing. Run: pip install python-docx")

    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)
    logger.info("DOCX extracted: %d characters from %d paragraphs", len(text), len(paragraphs))
    return text


def parse_resume(file_bytes: bytes, filename: str) -> str:
    """
    Dispatch to the right extractor based on file extension.
    Returns raw text (not yet cleaned — call clean_text() after this).

    Args:
        file_bytes: raw bytes from st.file_uploader or open(..., "rb")
        filename: original filename (used to infer extension)

    Returns:
        Raw text string (may contain unicode, multiple spaces, etc.)
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(file_bytes)
    elif ext == ".txt":
        return file_bytes.decode("utf-8", errors="replace")
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            "Please upload a PDF, DOCX, or plain-text (.txt) file."
        )
