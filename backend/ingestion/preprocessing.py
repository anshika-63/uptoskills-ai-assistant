
import re
import unicodedata
from collections import Counter





def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters."""
    return unicodedata.normalize("NFKC", text)


def clean_text(text: str) -> str:
    """Remove unwanted control characters."""
    text = normalize_unicode(text)
    return "".join(
        char for char in text
        if unicodedata.category(char) != "Cc" or char in "\n\t"
    )


def normalize_whitespace(text: str) -> str:
    """Normalize spaces and blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def preprocess_text(text: str) -> str:
    """Apply basic text preprocessing."""
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    return normalize_whitespace(clean_text(text))


def remove_repeated_headers_footers(
    pages: list,
    threshold: float = 0.6,
    lines_to_check: int = 3
) -> list:
    """Remove repeated lines appearing at page boundaries."""

    if not pages:
        return []

    if not 0 < threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    if lines_to_check < 1:
        raise ValueError("lines_to_check must be at least 1.")

    for page in pages:
        if not isinstance(page.get("text"), str):
            raise ValueError("Every page must contain text as a string.")

    if len(pages) < 2:
        return [dict(page) for page in pages]

    header_counts = Counter()
    footer_counts = Counter()

    for page in pages:
        lines = page["text"].splitlines()
        non_empty = [line.strip() for line in lines if line.strip()]

        header_counts.update(set(non_empty[:lines_to_check]))
        footer_counts.update(set(non_empty[-lines_to_check:]))

    min_occurrences = max(2, int(len(pages) * threshold + 0.999999))

    repeated_headers = {
        line for line, count in header_counts.items()
        if count >= min_occurrences
    }

    repeated_footers = {
        line for line, count in footer_counts.items()
        if count >= min_occurrences
    }

    cleaned_pages = []

    for page in pages:
        lines = page["text"].splitlines()
        non_empty_indices = [
            i for i, line in enumerate(lines) if line.strip()
        ]

        header_indices = set(non_empty_indices[:lines_to_check])
        footer_indices = set(non_empty_indices[-lines_to_check:])

        cleaned_lines = []

        for i, line in enumerate(lines):
            value = line.strip()

            is_header = i in header_indices and value in repeated_headers
            is_footer = i in footer_indices and value in repeated_footers

            if not (is_header or is_footer):
                cleaned_lines.append(line)

        cleaned_pages.append({
            **page,
            "text": "\n".join(cleaned_lines).strip()
        })

    return cleaned_pages

def verify_chunk_tokens(chunks: list, max_tokens: int = 500) -> list:
    """Estimate token count and verify chunk limits."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")

    verified_chunks = []

    for chunk in chunks:
        text = chunk["text"] if isinstance(chunk, dict) else chunk

        estimated_tokens = len(re.findall(r"\w+|[^\w\s]", text))

        item = chunk.copy() if isinstance(chunk, dict) else {"text": text}
        item["estimated_tokens"] = estimated_tokens
        item["within_token_limit"] = estimated_tokens <= max_tokens

        verified_chunks.append(item)

    return verified_chunks

def recursive_chunking(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> list:
    """Split text into sentence-aware chunks with word-safe overlap."""

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    text = preprocess_text(text)

    if not text:
        return []

    units = []

    for paragraph in text.splitlines():
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            if len(sentence) <= chunk_size:
                units.append(sentence)
                continue

            words = sentence.split()
            current_unit = ""

            for word in words:
                if len(word) > chunk_size:
                    if current_unit:
                        units.append(current_unit)
                        current_unit = ""

                    for i in range(0, len(word), chunk_size):
                        units.append(word[i:i + chunk_size])

                    continue

                candidate = (
                    current_unit + " " + word
                    if current_unit else word
                )

                if len(candidate) <= chunk_size:
                    current_unit = candidate
                else:
                    units.append(current_unit)
                    current_unit = word

            if current_unit:
                units.append(current_unit)

    chunks = []
    current = ""

    for unit in units:
        candidate = f"{current} {unit}".strip() if current else unit

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)

        overlap_words = []
        overlap_length = 0

        for word in reversed(current.split()):
            extra_length = len(word) + (
                1 if overlap_words else 0
            )

            if overlap_length + extra_length > overlap:
                break

            overlap_words.insert(0, word)
            overlap_length += extra_length

        prefix = " ".join(overlap_words)
        current = f"{prefix} {unit}".strip() if prefix else unit

        if len(current) > chunk_size:
            current = unit

    if current:
        chunks.append(current)

    return chunks

def attach_chunk_metadata(
    chunks: list,
    document_id: str,
    filename: str,
    file_type: str,
    page_number: int = None
) -> list:
    """Attach document metadata to every chunk."""

    total_chunks = len(chunks)
    result = []

    for index, chunk in enumerate(chunks):
        result.append({
            "text": chunk,
            "document_id": document_id,
            "filename": filename,
            "file_type": file_type,
            "page_number": page_number,
            "chunk_index": index,
            "total_chunks": total_chunks
        })

    return result
def verify_chunk_tokens(
    chunks: list,
    max_tokens: int = 500
) -> list:
    """Estimate token count and verify chunk limits."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")

    verified_chunks = []

    for chunk in chunks:
        text = chunk["text"] if isinstance(chunk, dict) else chunk

        estimated_tokens = len(re.findall(r"\w+|[^\w\s]", text))

        verified_chunks.append({
            **(chunk if isinstance(chunk, dict) else {"text": text}),
            "estimated_tokens": estimated_tokens,
            "within_token_limit": estimated_tokens <= max_tokens
        })

    return verified_chunks
def verify_chunk_tokens(chunks: list, max_tokens: int = 500) -> list:
    """Estimate token count and verify chunk limits."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")

    verified_chunks = []

    for chunk in chunks:
        text = chunk["text"] if isinstance(chunk, dict) else chunk

        estimated_tokens = len(re.findall(r"\w+|[^\w\s]", text))

        item = chunk.copy() if isinstance(chunk, dict) else {"text": text}
        item["estimated_tokens"] = estimated_tokens
        item["within_token_limit"] = estimated_tokens <= max_tokens

        verified_chunks.append(item)

    return verified_chunks