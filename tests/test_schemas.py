import pytest
from pydantic import ValidationError

from api.schemas import ChunkPreview, QueryRequest, QueryResponse, SourceCitation


def test_query_request_validates_top_k_bounds():
    QueryRequest(query="What is a relief valve?", top_k=1)
    QueryRequest(query="What is a relief valve?", top_k=10)

    with pytest.raises(ValidationError):
        QueryRequest(query="What is a relief valve?", top_k=0)

    with pytest.raises(ValidationError):
        QueryRequest(query="What is a relief valve?", top_k=11)


def test_query_response_accepts_realistic_payload_with_refused():
    response = QueryResponse(
        query="What is a relief valve?",
        answer="A relief valve protects systems from excess pressure.",
        sources=[
            SourceCitation(
                source="doe_hdbk_1018_v2_mechanical_science.pdf",
                page=61,
            )
        ],
        chunks_used=1,
        chunks=[
            ChunkPreview(
                text="Relief valves are used in liquid systems.",
                source="doe_hdbk_1018_v2_mechanical_science.pdf",
                page=61,
                retrieval_method="hybrid",
                rrf_score=0.032,
                rerank_score=None,
            )
        ],
        prompt_tokens=1200,
        completion_tokens=120,
        cost_usd=0.00025,
        latency_ms=2400,
        model="gpt-4o-mini",
        provider="openai",
        refused=False,
    )

    assert response.refused is False
    assert response.sources[0].page == 61
