import database
import rag_pipeline


def test_database_crud_roundtrip(client):
    gap = database.create_knowledge_gap(question="What is the parking fee?", context="no docs matched")
    assert gap["status"] == "open"
    assert gap["question"] == "What is the parking fee?"
    assert gap["resolved_entry_id"] is None

    fetched = database.get_knowledge_gap(gap["id"])
    assert fetched["id"] == gap["id"]

    updated = database.update_knowledge_gap_text(gap["id"], "What is the daily parking fee?", "still no docs")
    assert updated["question"] == "What is the daily parking fee?"
    assert updated["context"] == "still no docs"

    entry = database.create_entry({"name": "Parking Office", "type": "facility"})
    resolved = database.resolve_knowledge_gap(gap["id"], entry["id"])
    assert resolved["status"] == "resolved"
    assert resolved["resolved_entry_id"] == entry["id"]

    other = database.create_knowledge_gap(question="Another question")
    dismissed = database.dismiss_knowledge_gap(other["id"])
    assert dismissed["status"] == "dismissed"

    assert database.get_knowledge_gap("does-not-exist") is None
    assert database.update_knowledge_gap_text("does-not-exist", "x", "y") is None
    assert database.resolve_knowledge_gap("does-not-exist", "entry-1") is None
    assert database.dismiss_knowledge_gap("does-not-exist") is None


def test_get_knowledge_gaps_pagination_and_status_filter(client):
    for i in range(3):
        database.create_knowledge_gap(question=f"Open question {i}")
    dismissed_gap = database.create_knowledge_gap(question="Dismissed question")
    database.dismiss_knowledge_gap(dismissed_gap["id"])

    open_gaps, open_total = database.get_knowledge_gaps(status="open")
    assert open_total == 3
    assert all(g["status"] == "open" for g in open_gaps)

    dismissed_gaps, dismissed_total = database.get_knowledge_gaps(status="dismissed")
    assert dismissed_total == 1

    all_gaps, all_total = database.get_knowledge_gaps()
    assert all_total == 4

    page1, total = database.get_knowledge_gaps(page=1, per_page=2)
    assert total == 4
    assert len(page1) == 2


def test_list_get_patch_gaps_api(client):
    gap = database.create_knowledge_gap(question="What is the shuttle schedule?", context="no near matches")

    list_resp = client.get("/api/knowledge-gaps", params={"status": "open"})
    assert list_resp.status_code == 200
    data = list_resp.json()
    assert data["total"] == 1
    assert data["gaps"][0]["id"] == gap["id"]

    get_resp = client.get(f"/api/knowledge-gaps/{gap['id']}")
    assert get_resp.status_code == 200
    assert get_resp.json()["question"] == "What is the shuttle schedule?"

    assert client.get("/api/knowledge-gaps/does-not-exist").status_code == 404

    patch_resp = client.patch(
        f"/api/knowledge-gaps/{gap['id']}",
        json={"question": "What is the campus shuttle schedule?", "context": "updated context"},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["question"] == "What is the campus shuttle schedule?"

    assert client.patch(
        "/api/knowledge-gaps/does-not-exist", json={"question": "x", "context": "y"}
    ).status_code == 404


def test_resolve_gap_creates_entry_and_syncs_to_chroma(client):
    gap = database.create_knowledge_gap(question="What is the guest parking fee?", context="no docs found")

    resolve_resp = client.post(
        f"/api/knowledge-gaps/{gap['id']}/resolve",
        json={
            "question": "What is the guest parking fee?",
            "answer": "Guest parking is $5/day at the visitor lot.",
            "type": "facility",
            "location": "Visitor Lot",
        },
    )
    assert resolve_resp.status_code == 200
    body = resolve_resp.json()
    assert body["gap"]["status"] == "resolved"
    assert body["gap"]["resolved_entry_id"] == body["entry"]["id"]
    assert body["entry"]["description"] == "Guest parking is $5/day at the visitor lot."

    entry_resp = client.get(f"/api/entries/{body['entry']['id']}")
    assert entry_resp.status_code == 200

    ask_resp = client.post("/ask", json={"question": "How much is guest parking per day?"})
    source_ids = [s["id"] for s in ask_resp.json()["sources"]]
    assert body["entry"]["id"] in source_ids

    # Already-resolved gaps cannot be resolved again.
    second_resolve = client.post(
        f"/api/knowledge-gaps/{gap['id']}/resolve",
        json={"question": "x", "answer": "y", "type": "facility"},
    )
    assert second_resolve.status_code == 400


def test_resolve_gap_defaults_name_to_question_when_blank(client):
    gap = database.create_knowledge_gap(question="What is the gym's weekend schedule?")

    resolve_resp = client.post(
        f"/api/knowledge-gaps/{gap['id']}/resolve",
        json={
            "question": "What is the gym's weekend schedule?",
            "answer": "Open 8am-8pm on weekends.",
            "type": "facility",
        },
    )
    assert resolve_resp.status_code == 200
    assert resolve_resp.json()["entry"]["name"] == "What is the gym's weekend schedule?"


def test_dismiss_gap_api(client):
    gap = database.create_knowledge_gap(question="Irrelevant off-topic question")

    dismiss_resp = client.post(f"/api/knowledge-gaps/{gap['id']}/dismiss")
    assert dismiss_resp.status_code == 200
    assert dismiss_resp.json()["status"] == "dismissed"

    open_gaps, open_total = database.get_knowledge_gaps(status="open")
    assert open_total == 0

    assert client.post("/api/knowledge-gaps/does-not-exist/dismiss").status_code == 404


def test_empty_retrieval_logs_gap_without_calling_llm(client, mock_call_llm, monkeypatch):
    empty_results = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
        "ids": [[]],
    }
    monkeypatch.setattr(rag_pipeline, "query_knowledge_base", lambda collection, query, top_k: empty_results)

    resp = client.post("/ask", json={"question": "What is the meaning of life on this campus?"})
    assert resp.status_code == 200
    mock_call_llm.assert_not_called()

    gaps, total = database.get_knowledge_gaps(status="open")
    assert total == 1
    assert gaps[0]["question"] == "What is the meaning of life on this campus?"
    assert "No documents were retrieved" in gaps[0]["context"]
