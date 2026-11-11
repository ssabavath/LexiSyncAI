
"""PDF ingestion with page-aware and article-aware chunking."""

import re
import fitz


# Recognise genuine legal article and section headings.
ARTICLE_RE = re.compile(
    r"^\s*(Article\s+\d+[a-z]?)\s*$",
    re.IGNORECASE
)

SECTION_RE = re.compile(
    r"^\s*(CHAPTER|SECTION)\s+[IVXLCDM0-9]+\s*$",
    re.IGNORECASE
)

PARAGRAPH_RE = re.compile(
    r"^\s*(\d{1,2})\.\s*$"
)


def clean_text(text):
    """Clean extracted PDF text."""

    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_into_chunks(text, chunk_size=1100, overlap=180):
    """Split long text while retaining a small overlap."""

    text = clean_text(text)

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        if end < len(text):

            boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def process_pdf(pdf_path: str, document_name: str):
    """
    Extract page-level evidence while preserving article headings
    across PDF pages.
    """

    doc = fitz.open(pdf_path)

    chunks = []
    warnings = []

    active_section = "Page text"
    active_paragraph = None

    for page_index, page in enumerate(doc):

        page_number = page_index + 1

        text = page.get_text("text", sort=True)

        if len(text.strip()) < 30:

            warnings.append(
                f"{document_name}, page {page_number}: "
                "Very little extractable text; OCR may be needed."
            )

            continue

        lines = text.splitlines()

        current_lines = []

        def save_section():

            if not current_lines:
                return

            section_text = clean_text(
                "\n".join(current_lines)
            )

            if not section_text:
                return

            # Keep article and paragraph information in metadata.
            for chunk_number, chunk_text in enumerate(
                split_into_chunks(section_text),
                start=1
            ):

                chunks.append({
                    "chunk_id": (
                        f"{document_name}-p{page_number}-"
                        f"c{len(chunks) + 1}"
                    ),
                    "document": document_name,
                    "page": page_number,
                    "section": active_section,
                    "paragraph": active_paragraph,
                    "text": chunk_text
                })

        for line in lines:

            stripped = line.strip()

            if not stripped:
                current_lines.append(line)
                continue

            # Detect genuine article headings.
            article_match = ARTICLE_RE.match(stripped)

            if article_match:

                save_section()

                active_section = article_match.group(1).title()
                active_paragraph = None

                current_lines = [line]

                continue

            # Detect chapter or section headings.
            section_match = SECTION_RE.match(stripped)

            if section_match:

                save_section()

                active_section = stripped
                active_paragraph = None

                current_lines = [line]

                continue

            # Track numbered paragraphs, e.g. 1. or 2.
            paragraph_match = PARAGRAPH_RE.match(stripped)

            if paragraph_match:

                active_paragraph = paragraph_match.group(1)

            # Normal text continues under the current article.
            current_lines.append(line)

        # Save page content without resetting active_section.
        save_section()

    doc.close()

    return chunks, warnings
