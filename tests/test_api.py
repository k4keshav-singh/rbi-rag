"""The HTTP surface.

Runs the real request path -- routing, validation, retrieval, generation,
citation checking, cost accounting -- against fakes. No database, no network,
no API key.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from rbi_rag.api.app import create_app
from rbi_rag.config import Settings
from rbi_rag.index.store import ChunkStore
from rbi_rag.schemas import AnswerPayload, CitationClaim
from tests.fakes import CHUNK_SET, FakeEmbedder, FakeLLM, FakeReranker, build_store

GOOD_QUOTE = "nine per cent on an ongoing basis"


def good_payload() -> AnswerPayload:
    return AnswerPayload(
        answer="Banks must maintain a CRAR of nine per cent.",
        citations=[CitationClaim(chunk_id="c-capital", quote=GOOD_QUOTE)],
        sufficient_evidence=True,
    )


def make_client(
    llm: FakeLLM | None = None,
    *,
    store_fails: bool = False,
    **setting_overrides: object,
) -> TestClient:
    settings = Settings(_env_file=None, retriever="dense", **setting_overrides)  # type: ignore[call-arg]
    store = build_store()

    @contextmanager
    def provider() -> Iterator[ChunkStore]:
        if store_fails:
            raise psycopg.OperationalError("database is down")
        yield store

    app = create_app(
        settings=settings,
        llm=llm or FakeLLM([good_payload()]),
        store_provider=provider,
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        chunk_set_resolver=lambda: CHUNK_SET,
    )
    return TestClient(app)


class TestHealth:
    def test_reports_ok_when_everything_is_wired(self) -> None:
        with make_client() as client:
            body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["checks"]["database"] == "ok"

    def test_reports_which_dependency_is_down(self) -> None:
        """A monitor needs to know what broke, not just that something did."""
        with make_client(store_fails=True) as client:
            response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert "unavailable" in body["checks"]["database"]

    def test_reports_the_price_table_date(self) -> None:
        with make_client() as client:
            checks = client.get("/health").json()["checks"]
        assert checks["price_table_date"]
        assert checks["model_priced"] is True

    def test_never_echoes_the_api_key(self) -> None:
        with make_client() as client:
            body = client.get("/health").text
        assert "api_key_configured" in body
        assert "sk-ant" not in body


class TestQuery:
    def test_answers_with_citations(self) -> None:
        with make_client() as client:
            body = client.post("/query", json={"question": "what capital ratio?"}).json()
        assert body["abstained"] is False
        assert body["citations"][0]["chunk_id"] == "c-capital"
        assert body["citations"][0]["quote"] == GOOD_QUOTE

    def test_response_carries_the_contract_fields(self) -> None:
        with make_client() as client:
            body = client.post("/query", json={"question": "what capital ratio?"}).json()
        for field in (
            "answer",
            "citations",
            "retrieved_chunk_ids",
            "latency_ms",
            "usage",
            "cost_usd",
            "model",
            "retriever",
        ):
            assert field in body, field

    def test_reports_cost_and_tokens(self) -> None:
        with make_client() as client:
            body = client.post("/query", json={"question": "what capital ratio?"}).json()
        assert body["usage"]["input_tokens"] == 1000
        assert body["cost_usd"] > 0

    def test_cost_is_null_for_an_unpriced_model(self) -> None:
        llm = FakeLLM([good_payload()], model="some-unreleased-model")
        with make_client(llm) as client:
            body = client.post("/query", json={"question": "what capital ratio?"}).json()
        assert body["cost_usd"] is None

    def test_latency_is_measured(self) -> None:
        with make_client() as client:
            body = client.post("/query", json={"question": "what capital ratio?"}).json()
        assert body["latency_ms"] > 0

    def test_abstains_rather_than_answering_unsupported(self) -> None:
        payload = AnswerPayload(answer="", citations=[], sufficient_evidence=False)
        with make_client(FakeLLM([payload])) as client:
            body = client.post("/query", json={"question": "what colour is the sky?"}).json()
        assert body["abstained"] is True
        assert body["citations"] == []

    def test_a_fabricated_citation_never_reaches_the_client(self) -> None:
        bad = AnswerPayload(
            answer="Banks must maintain twelve per cent.",
            citations=[CitationClaim(chunk_id="c-capital", quote="twelve per cent at all times")],
            sufficient_evidence=True,
        )
        with make_client(FakeLLM([bad, bad])) as client:
            body = client.post("/query", json={"question": "what capital ratio?"}).json()
        assert body["abstained"] is True
        assert body["citations"] == []
        assert "twelve per cent" not in body["answer"]

    def test_the_retriever_can_be_overridden_per_request(self) -> None:
        with make_client() as client:
            body = client.post(
                "/query", json={"question": "capital adequacy", "retriever": "fts"}
            ).json()
        assert body["retriever"] == "fts"

    def test_top_k_is_honoured(self) -> None:
        with make_client() as client:
            body = client.post(
                "/query", json={"question": "what capital ratio?", "top_k": 1}
            ).json()
        assert len(body["retrieved_chunk_ids"]) == 1

    @pytest.mark.parametrize(
        "payload",
        [{}, {"question": ""}, {"question": "x", "top_k": 0}, {"question": "x", "top_k": 999}],
    )
    def test_bad_requests_are_rejected(self, payload: dict[str, object]) -> None:
        with make_client() as client:
            assert client.post("/query", json=payload).status_code == 422

    def test_a_database_outage_is_a_503_not_an_unhandled_500(self) -> None:
        with make_client(store_fails=True) as client:
            response = client.post("/query", json={"question": "what capital ratio?"})
        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"]

    def test_every_response_carries_a_request_id(self) -> None:
        with make_client() as client:
            response = client.post("/query", json={"question": "what capital ratio?"})
        assert response.headers["x-request-id"]


class TestFrontend:
    def test_the_page_is_served(self) -> None:
        with make_client() as client:
            response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")

    def test_it_renders_the_pieces_the_brief_asks_for(self) -> None:
        """Answer, clickable citations, latency and cost."""
        with make_client() as client:
            page = client.get("/").text
        for marker in ("citations", "latency_ms", "cost_usd", "abstained"):
            assert marker in page, marker

    def test_citations_deep_link_to_the_pdf_page(self) -> None:
        with make_client() as client:
            page = client.get("/").text
        assert "#page=" in page

    def test_it_pulls_in_nothing_from_a_cdn(self) -> None:
        """One self-contained file: no build step, no third-party fetch."""
        with make_client() as client:
            page = client.get("/").text
        assert "http://" not in page.replace("http://127.0.0.1", "")
        assert "cdn" not in page.lower()

    def test_a_missing_page_does_not_take_the_api_down(self, tmp_path: Path) -> None:
        """A packaging slip must not look like a total outage."""
        from rbi_rag.api import app as app_module

        original = app_module.STATIC_DIR
        app_module.STATIC_DIR = tmp_path / "absent"
        try:
            with make_client() as client:
                assert client.get("/").status_code == 200
                assert client.get("/health").status_code == 200
        finally:
            app_module.STATIC_DIR = original
