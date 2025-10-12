# preprocessing.py
# Read PDFs in data/ and produce chunks.json
# Preserves section numbers and headings

import pdfplumber
import json
from pathlib import Path
import re

DATA_DIR = Path("data")
OUT_FILE = Path("chunks.json")
CHUNK_SIZE = 1200  # characters per chunk (tune as needed)


def normalize_whitespace(s: str) -> str:
    """Preserve line breaks for headings, but remove extra spaces."""
    # Replace multiple spaces with single space
    s = re.sub(r"[ \t]+", " ", s)
    # Normalize newlines (keep single line break)
    s = re.sub(r"\n+", "\n", s)
    return s.strip()


all_chunks = []

for pdf in DATA_DIR.glob("*.pdf"):
    print(f"Processing {pdf.name}...")
    with pdfplumber.open(pdf) as f:
        text_pages = []
        for page in f.pages:
            page_text = page.extract_text()
            if page_text:
                text_pages.append(page_text)

        full_text = "\n".join(text_pages)
        full_text = normalize_whitespace(full_text)
        if not full_text:
            continue

        # Chunk by characters, but try to split at line breaks near the chunk boundary
        start = 0
        while start < len(full_text):
            end = start + CHUNK_SIZE
            if end < len(full_text):
                # Move end to nearest newline to avoid cutting a sentence or section
                newline_pos = full_text.rfind("\n", start, end)
                if newline_pos > start:
                    end = newline_pos
            chunk = full_text[start:end].strip()
            all_chunks.append({"text": chunk, "source": pdf.name})
            start = end

print(f"Total chunks created: {len(all_chunks)}")

with OUT_FILE.open("w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print(f"Saved chunks to {OUT_FILE}")
