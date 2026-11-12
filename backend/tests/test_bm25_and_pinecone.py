import os
import pytest
from backend.app.services.bm25_service import BM25Service
from backend.app.services.pinecone_service import PineconeService

def test_bm25_fit_encode_and_persistence(tmp_path):
    params_file = str(tmp_path / "test_bm25_params.json")
    bm25 = BM25Service(params_path=params_file)

    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence and retrieval augmented generation are transforming search.",
        "Pinecone provides serverless vector database capabilities with hybrid search.",
        "Python FastAPI backend with Redis cache and cross-encoder reranker."
    ]

    bm25.fit_and_save(corpus)
    assert os.path.exists(params_file)
    assert os.path.getsize(params_file) > 0

    # Test document encoding
    doc_encodings = bm25.encode_documents(corpus[:2])
    assert len(doc_encodings) == 2
    for enc in doc_encodings:
        assert "indices" in enc
        assert "values" in enc
        assert len(enc["indices"]) == len(enc["values"])

    # Test query encoding
    q_enc = bm25.encode_query("retrieval augmented generation")
    assert "indices" in q_enc
    assert "values" in q_enc
    assert len(q_enc["indices"]) > 0

    # Test re-loading from saved params
    reloaded_bm25 = BM25Service(params_path=params_file)
    reloaded_q = reloaded_bm25.encode_query("retrieval augmented generation")
    assert reloaded_q["indices"] == q_enc["indices"]

def test_pinecone_service_hybrid_and_delete():
    pinecone = PineconeService()
    # Ensure it runs in local mock mode for unit tests
    pinecone._init_local_mock()
    assert pinecone.is_using_mock()

    test_vectors = [
        {
            "id": "doc1#chunk-0",
            "values": [0.1] * 768,
            "sparse_values": {"indices": [10, 20], "values": [1.0, 2.0]},
            "metadata": {"doc_id": "doc1", "file_name": "test1.pdf", "page": 1, "text": "First chunk content"}
        },
        {
            "id": "doc1#chunk-1",
            "values": [0.2] * 768,
            "sparse_values": {"indices": [20, 30], "values": [2.0, 3.0]},
            "metadata": {"doc_id": "doc1", "file_name": "test1.pdf", "page": 2, "text": "Second chunk content"}
        },
        {
            "id": "doc2#chunk-0",
            "values": [-0.1] * 768,
            "sparse_values": {"indices": [99], "values": [0.5]},
            "metadata": {"doc_id": "doc2", "file_name": "test2.pdf", "page": 1, "text": "Another document chunk"}
        }
    ]

    pinecone.upsert_vectors(test_vectors)
    assert len(pinecone._mock_vectors) == 3

    # Test hybrid query
    matches = pinecone.hybrid_query(
        dense_vector=[0.1] * 768,
        sparse_vector={"indices": [10, 20], "values": [1.0, 2.0]},
        top_k=2,
        alpha=0.5
    )
    assert len(matches) == 2
    assert matches[0]["id"] == "doc1#chunk-0" or matches[0]["id"] == "doc1#chunk-1"

    # Test delete by doc_id prefix
    pinecone.delete_by_doc_id("doc1", chunk_ids=["doc1#chunk-0", "doc1#chunk-1"])
    assert "doc1#chunk-0" not in pinecone._mock_vectors
    assert "doc1#chunk-1" not in pinecone._mock_vectors
    assert "doc2#chunk-0" in pinecone._mock_vectors
