from app.chunking import chunk_corpus


def test_chunks_cover_all_files_with_unique_ids(corpus_dir):
    chunks = chunk_corpus(corpus_dir)
    assert len({c.source for c in chunks}) == 6
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))          # ids are unique
    assert all("#" in cid for cid in ids)      # "<stem>#<slug>" shape


def test_lease_rent_chunk_carries_the_parties(corpus_dir):
    # Gold Q14 needs the parties (front-matter) and the rent (a section) together.
    chunks = chunk_corpus(corpus_dir)
    rent = next(
        c for c in chunks
        if c.source == "06_property_lease_clause.md" and "rent" in c.section_header.lower()
    )
    assert "Kiran Patel" in rent.text     # lessor, from the front-matter
    assert "₹45,000" in rent.text          # rent fact, UTF-8 preserved


def test_file_without_headers_is_one_chunk(corpus_dir):
    # The statute uses bold "Section" labels, not "##", so it stays whole.
    chunks = chunk_corpus(corpus_dir)
    statute = [c for c in chunks if c.source == "04_statute_style_excerpt_fictional.md"]
    assert len(statute) == 1
    assert "9%" in statute[0].text and "30 days" in statute[0].text


def test_chunk_id_is_deterministic(corpus_dir):
    assert [c.chunk_id for c in chunk_corpus(corpus_dir)] == \
           [c.chunk_id for c in chunk_corpus(corpus_dir)]


def test_duplicate_section_headers_stay_unique(tmp_path):
    from app.chunking import chunk_markdown

    path = tmp_path / "dup.md"
    path.write_text("# Title\n\n## Notes\nfirst\n\n## Notes\nsecond\n", encoding="utf-8")
    ids = [c.chunk_id for c in chunk_markdown(path)]
    assert ids == ["dup#notes", "dup#notes-2"]   # no overwrite on re-ingest
