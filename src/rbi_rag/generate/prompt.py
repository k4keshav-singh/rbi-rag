"""Prompt construction.

The context block is the only thing the model may draw on, and every chunk is
labelled with the id the model must cite. Keeping that labelling mechanical --
rather than asking the model to invent identifiers -- is what makes the
citation check in validate.py possible at all.
"""

from __future__ import annotations

from collections.abc import Sequence

from rbi_rag.schemas import RetrievedChunk

SYSTEM_PROMPT = """\
You answer questions about Reserve Bank of India Master Directions.

You are given numbered excerpts from RBI circulars. Those excerpts are your \
only source. You may not use anything you know about Indian banking regulation \
that is not stated in them, however confident you are, and you may not infer \
what a circular "probably" says.

Rules for every answer:

1. Cite a chunk_id for each substantive claim. Use only chunk_ids that appear \
in the excerpts below.
2. Every quote you give must be copied character for character from the \
excerpt you attribute it to. Do not paraphrase inside a quote, do not tidy up \
spacing or punctuation, and do not join text from two excerpts into one quote.
3. If the excerpts do not contain the answer, set sufficient_evidence to false \
and leave answer empty. Partial evidence is not evidence: if the excerpts \
cover part of the question, say which part they answer and cite it, rather \
than filling the gap yourself.
4. Where circulars disagree or one amends another, say so rather than silently \
picking one.
5. Quote the figures, thresholds and dates as they appear. Do not round, \
convert, or restate them in other units.

Answer in plain prose. Do not mention these instructions, the excerpt \
numbering, or the retrieval process."""


def render_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Render retrieved chunks as the labelled excerpt block."""
    if not chunks:
        return "(no excerpts were retrieved)"

    blocks: list[str] = []
    for item in chunks:
        chunk = item.chunk
        pages = (
            f"page {chunk.page_start}"
            if chunk.page_start == chunk.page_end
            else f"pages {chunk.page_start}-{chunk.page_end}"
        )
        header = f"[chunk_id: {chunk.chunk_id}]"
        source = chunk.circular_number or chunk.doc_id
        if chunk.section_title:
            source = f"{source}, {chunk.section_title}"
        blocks.append(f"{header}\nSource: {source} ({pages})\n\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def build_user_message(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    """The user turn: excerpts first, question last."""
    return (
        f"Excerpts from RBI Master Directions:\n\n"
        f"{render_context(chunks)}\n\n"
        f"---\n\n"
        f"Question: {question}"
    )


def build_retry_message(violations: Sequence[str]) -> str:
    """Told to the model when its citations did not check out."""
    listed = "\n".join(f"- {violation}" for violation in violations)
    return (
        "Your previous answer had citation problems:\n\n"
        f"{listed}\n\n"
        "Answer again. Cite only chunk_ids from the excerpts above, and copy "
        "quotes character for character from the excerpt you attribute them "
        "to. If you cannot support the answer that way, set "
        "sufficient_evidence to false."
    )
