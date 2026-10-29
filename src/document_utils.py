from __future__ import annotations

import io
import re
from typing import BinaryIO


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    tokens = cleaned.split()
    merged: list[str] = []
    for token in tokens:
        if len(token) == 1 and merged and merged[-1].isalpha() and len(merged[-1]) > 2:
            merged[-1] += token
        else:
            merged.append(token)
    return " ".join(merged)


def extract_document_text(file_obj: BinaryIO) -> str:
    """Extract readable text from uploaded PDF, image, or text files."""
    if file_obj is None:
        raise ValueError("No file was uploaded.")

    name = getattr(file_obj, "name", "").lower()
    file_obj.seek(0)
    data = file_obj.read()
    if not data:
        raise ValueError("The uploaded file is empty.")

    if name.endswith(".txt"):
        try:
            return _normalize_text(data.decode("utf-8-sig"))
        except UnicodeDecodeError:
            return _normalize_text(data.decode("latin-1", errors="ignore"))

    if name.endswith(".pdf"):
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PDF extraction requires PyMuPDF. Install it with pip install pymupdf.") from exc

        doc = fitz.open(stream=data, filetype="pdf")
        extracted_pages = [page.get_text("text") for page in doc]
        doc.close()
        text = "\n".join(page for page in extracted_pages if page and page.strip())
        normalized = _normalize_text(text)
        if not normalized:
            raise ValueError("No readable text could be extracted from the PDF.")
        return normalized

    if name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif")):
        try:
            import numpy as np
            from PIL import Image
            from rapidocr_onnxruntime import RapidOCR

            try:
                import pytesseract
            except ImportError:
                pytesseract = None
        except ImportError as exc:
            raise RuntimeError(
                "Image OCR requires Pillow plus an OCR engine. Install the dependencies in requirements.txt."
            ) from exc

        try:
            img = Image.open(io.BytesIO(data))
            img.load()
            text = ""
            if pytesseract is not None:
                try:
                    text = pytesseract.image_to_string(img)
                except Exception:
                    text = ""
            if not text.strip():
                result, _ = RapidOCR()(np.asarray(img))
                text = " ".join(item[1] for item in (result or []) if item and len(item) > 1)
        except Exception as exc:  # pragma: no cover - environment-specific runtime error path
            raise RuntimeError(f"Could not read text from the uploaded image: {exc}") from exc

        normalized = _normalize_text(text)
        if not normalized:
            raise ValueError("No text could be recognized in the uploaded image.")
        return normalized

    raise ValueError("Unsupported file type. Upload a PDF, image, or text file.")
