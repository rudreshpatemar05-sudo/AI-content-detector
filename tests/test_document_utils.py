from io import BytesIO

import pytest

from src.document_utils import extract_document_text


def _make_pdf_bytes(text: str) -> BytesIO:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


def test_extract_document_text_from_pdf():
    text = "This is a sample human-written paragraph for the detector."
    buffer = _make_pdf_bytes(text)
    buffer.name = "sample.pdf"

    extracted = extract_document_text(buffer)

    assert text in extracted
