"""Header-aware chunking for the markdown corpus.

Each file is split into one chunk per ``##`` section. The document title and any
front-matter (the bold lines before the first section) are prepended to every
chunk, so a single chunk carries enough context to be cited on its own — e.g. a
question about the rent *and* the parties can be answered from the lease's rent
section alone. Files without ``##`` headers (the statute and the counsel notes)
become a single chunk. No fixed-size windows or overlap: the sections are already
small and self-contained.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

_SECTION = re.compile(r"(?m)^##\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str          # "<file stem>#<section slug>" — stable, so re-ingest overwrites
    source: str            # file name, shown in citations
    doc_title: str
    section_header: str
    text: str              # contextual text: title + front-matter + section body
    content_hash: str


def _slug(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", header.lower()).strip("-") or "overview"


def chunk_markdown(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8").strip()
    title = next(
        (ln[2:].strip() for ln in raw.splitlines() if ln.startswith("# ")), path.stem
    )

    # re.split keeps the captured headers: [head, header1, body1, header2, body2, ...]
    parts = _SECTION.split(raw)
    head = parts[0].strip()
    front_matter = "\n".join(
        ln for ln in head.splitlines() if not ln.startswith("# ")
    ).strip()
    context = f"{title}\n{front_matter}".strip()

    if len(parts) == 1:
        # No "##" sections: keep the whole file as one chunk.
        sections = [("overview", raw)]
    else:
        sections = [
            (parts[i].strip(), parts[i + 1].strip())
            for i in range(1, len(parts), 2)
        ]

    chunks = []
    seen_slugs: dict[str, int] = {}
    for header, body in sections:
        if header == "overview":
            text = body  # already the full file (title + front-matter + body)
        else:
            text = f"{context}\n\n## {header}\n{body}".strip()
        slug = _slug(header)
        seen_slugs[slug] = seen_slugs.get(slug, 0) + 1
        if seen_slugs[slug] > 1:  # two headers in one file slug the same -> keep ids unique
            slug = f"{slug}-{seen_slugs[slug]}"
        chunks.append(
            Chunk(
                chunk_id=f"{path.stem}#{slug}",
                source=path.name,
                doc_title=title,
                section_header=header,
                text=text,
                content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
        )
    return chunks


def chunk_corpus(corpus_dir: Path) -> list[Chunk]:
    """Chunk every ``*.md`` file in the directory, sorted by name for stable ids."""
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        chunks.extend(chunk_markdown(path))
    return chunks
