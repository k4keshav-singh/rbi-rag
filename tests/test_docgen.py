"""The documentation generator.

Its job is to make rule 1 mechanical: a number reaches the README by being
measured, or it does not reach the README. Most of these tests are about the
TODO path, because that is the one that holds while there is nothing to report
and the one a hand-edit would quietly replace.
"""

from pathlib import Path

import pytest

from rbi_rag.docgen import (
    TODO,
    Decision,
    MarkerError,
    parse_decisions,
    render_csv_table,
    render_decision_index,
    render_results,
    splice,
    update_file,
)

RETRIEVAL_CSV = """\
arm,strategy,chunk_size,retriever,recall@5,mrr
section_aware-512-hybrid-norerank,section_aware,512,hybrid,0.8400,0.7100
section_aware-512-dense-norerank,section_aware,512,dense,0.8100,0.6900
section_aware-512-fts-norerank,section_aware,512,fts,,0.4200
"""


class TestRenderResults:
    def test_no_results_says_todo_and_why(self, tmp_path: Path) -> None:
        rendered = render_results(tmp_path)
        assert TODO in rendered
        assert "verified" in rendered

    def test_no_results_never_emits_a_number(self, tmp_path: Path) -> None:
        """An empty section must not be mistakable for a measurement."""
        rendered = render_results(tmp_path)
        assert not any(character.isdigit() for character in rendered.replace("rbi_rag", ""))

    def test_a_present_csv_becomes_a_table(self, tmp_path: Path) -> None:
        (tmp_path / "retrieval.csv").write_text(RETRIEVAL_CSV, encoding="utf-8")
        rendered = render_results(tmp_path)
        assert "| arm |" in rendered
        assert "section_aware-512-hybrid-norerank" in rendered
        assert "0.8400" in rendered

    def test_the_heading_names_the_file(self, tmp_path: Path) -> None:
        (tmp_path / "retrieval.csv").write_text(RETRIEVAL_CSV, encoding="utf-8")
        assert "### Retrieval" in render_results(tmp_path)

    def test_both_tables_appear_when_both_exist(self, tmp_path: Path) -> None:
        (tmp_path / "retrieval.csv").write_text(RETRIEVAL_CSV, encoding="utf-8")
        (tmp_path / "generation.csv").write_text("arm,faithfulness\nx,0.9\n", encoding="utf-8")
        rendered = render_results(tmp_path)
        assert "### Retrieval" in rendered
        assert "### Generation" in rendered

    def test_it_says_which_numbers_reproduce(self, tmp_path: Path) -> None:
        (tmp_path / "retrieval.csv").write_text(RETRIEVAL_CSV, encoding="utf-8")
        rendered = render_results(tmp_path)
        assert "deterministic" in rendered


class TestRenderCsvTable:
    def test_an_empty_metric_becomes_todo_not_a_blank(self, tmp_path: Path) -> None:
        """A blank markdown cell reads like a measurement of nothing."""
        path = tmp_path / "retrieval.csv"
        path.write_text(RETRIEVAL_CSV, encoding="utf-8")
        table = render_csv_table(path)
        assert f"| {TODO} |" in table

    def test_a_header_only_csv_says_so(self, tmp_path: Path) -> None:
        path = tmp_path / "retrieval.csv"
        path.write_text("arm,recall@5\n", encoding="utf-8")
        assert "no rows" in render_csv_table(path)

    def test_truncation_is_disclosed(self, tmp_path: Path) -> None:
        """Hiding arms from an ablation without saying so is exactly what a
        reader would be right to distrust."""
        path = tmp_path / "retrieval.csv"
        path.write_text(RETRIEVAL_CSV, encoding="utf-8")
        table = render_csv_table(path, max_rows=1)
        assert "Showing 1 of 3 rows" in table

    def test_every_row_is_shown_by_default(self, tmp_path: Path) -> None:
        path = tmp_path / "retrieval.csv"
        path.write_text(RETRIEVAL_CSV, encoding="utf-8")
        table = render_csv_table(path)
        assert "Showing" not in table
        assert table.count("section_aware-512") == 3


DECISIONS_TEXT = """\
# Decisions

## 0001 — Postgres with pgvector as the only datastore

**Date:** 2026-10-29 · **Phase:** 0

Body.

## 0014 — onnxruntime as the embedding backend, not sentence-transformers

**Date:** 2026-10-30 · **Phase:** 2

Body.
"""


class TestDecisionIndex:
    def test_entries_are_found(self) -> None:
        decisions = parse_decisions(DECISIONS_TEXT)
        assert [d.number for d in decisions] == ["0001", "0014"]

    def test_the_phase_is_picked_up(self) -> None:
        decisions = parse_decisions(DECISIONS_TEXT)
        assert decisions[0].phase == "0"
        assert decisions[1].phase == "2"

    def test_a_missing_phase_is_todo_not_blank(self) -> None:
        index = render_decision_index([Decision("0099", "Something", None)])
        assert TODO in index

    def test_the_index_links_to_each_entry(self) -> None:
        index = render_decision_index(parse_decisions(DECISIONS_TEXT))
        assert "(#0001--postgres-with-pgvector-as-the-only-datastore)" in index

    def test_anchors_match_githubs_slugging(self) -> None:
        """GitHub strips the em dash but keeps both spaces around it, so the
        anchor carries a double hyphen. Getting this wrong produces an index
        whose links fail silently."""
        anchor = Decision("0002", "A shared `schemas/` package", "0").anchor
        assert anchor == "0002--a-shared-schemas-package"

    def test_anchors_keep_underscores(self) -> None:
        anchor = Decision("0027", "Why section_aware wins", "4").anchor
        assert "section_aware" in anchor

    def test_no_decisions_says_so(self) -> None:
        assert "No decisions" in render_decision_index([])

    def test_the_real_file_parses(self) -> None:
        """Guards the regex against the actual document drifting."""
        from rbi_rag.config import PROJECT_ROOT

        decisions = parse_decisions((PROJECT_ROOT / "DECISIONS.md").read_text(encoding="utf-8"))
        assert len(decisions) >= 30
        assert all(d.phase is not None for d in decisions)


class TestSplice:
    def test_the_block_replaces_what_is_between_the_markers(self) -> None:
        out = splice("a\n<!-- B -->\nold\n<!-- E -->\nz", "<!-- B -->", "<!-- E -->", "new")
        assert "new" in out
        assert "old" not in out

    def test_the_markers_survive(self) -> None:
        out = splice("<!-- B -->\nold\n<!-- E -->", "<!-- B -->", "<!-- E -->", "new")
        assert out.startswith("<!-- B -->")
        assert out.endswith("<!-- E -->")

    def test_content_outside_the_markers_is_untouched(self) -> None:
        out = splice(
            "keep me\n<!-- B -->\nold\n<!-- E -->\nand me", "<!-- B -->", "<!-- E -->", "new"
        )
        assert out.startswith("keep me")
        assert out.endswith("and me")

    def test_missing_markers_are_an_error(self) -> None:
        with pytest.raises(MarkerError, match="missing or reversed"):
            splice("no markers here", "<!-- B -->", "<!-- E -->", "new")

    def test_reversed_markers_are_an_error(self) -> None:
        with pytest.raises(MarkerError, match="missing or reversed"):
            splice("<!-- E -->\n<!-- B -->", "<!-- B -->", "<!-- E -->", "new")

    def test_splicing_is_idempotent(self) -> None:
        """So `--check` does not report a stale file on every run."""
        once = splice("<!-- B -->\nold\n<!-- E -->", "<!-- B -->", "<!-- E -->", "new")
        twice = splice(once, "<!-- B -->", "<!-- E -->", "new")
        assert once == twice


class TestUpdateFile:
    def test_it_reports_a_change(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.md"
        path.write_text("<!-- B -->\nold\n<!-- E -->\n", encoding="utf-8")
        assert update_file(path, "<!-- B -->", "<!-- E -->", "new") is True
        assert "new" in path.read_text(encoding="utf-8")

    def test_it_reports_no_change_when_already_current(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.md"
        path.write_text("<!-- B -->\nold\n<!-- E -->\n", encoding="utf-8")
        update_file(path, "<!-- B -->", "<!-- E -->", "new")
        assert update_file(path, "<!-- B -->", "<!-- E -->", "new") is False


class TestTheRealDocuments:
    def test_the_readme_carries_the_results_markers(self) -> None:
        from rbi_rag.config import PROJECT_ROOT
        from rbi_rag.docgen import RESULTS_BEGIN, RESULTS_END

        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        assert RESULTS_BEGIN in readme
        assert RESULTS_END in readme

    def test_the_decisions_file_carries_the_index_markers(self) -> None:
        from rbi_rag.config import PROJECT_ROOT
        from rbi_rag.docgen import DECISIONS_BEGIN, DECISIONS_END

        decisions = (PROJECT_ROOT / "DECISIONS.md").read_text(encoding="utf-8")
        assert DECISIONS_BEGIN in decisions
        assert DECISIONS_END in decisions


class TestCrossDocumentLinks:
    """Every DECISIONS link in the prose must resolve.

    Worth a permanent test: the anchors are generated, the links are written
    by hand, and a mismatch between them fails silently in a renderer. The
    first version of the generator slugged headings differently from GitHub
    and every link in the repository was dead.
    """

    def _valid_anchors(self) -> set[str]:
        from rbi_rag.config import PROJECT_ROOT

        text = (PROJECT_ROOT / "DECISIONS.md").read_text(encoding="utf-8")
        return {decision.anchor for decision in parse_decisions(text)}

    def _links_in(self, relative: str) -> list[str]:
        import re

        from rbi_rag.config import PROJECT_ROOT

        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        return re.findall(r"DECISIONS\.md#([a-z0-9_-]+)", text)

    @pytest.mark.parametrize("document", ["README.md", "docs/architecture.md"])
    def test_every_decision_link_resolves(self, document: str) -> None:
        valid = self._valid_anchors()
        broken = [anchor for anchor in self._links_in(document) if anchor not in valid]
        assert not broken, f"{document} links to missing anchors: {broken}"

    def test_the_prose_actually_links_to_decisions(self) -> None:
        """Guards against the test above passing because nothing links at all."""
        assert len(self._links_in("README.md")) >= 5
