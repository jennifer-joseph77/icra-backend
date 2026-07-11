def create_session(client) -> str:
    resp = client.post("/sessions")
    assert resp.status_code == 200
    return resp.json()["session_id"]


def test_session_create_list_messages_delete_lifecycle(client):
    session_id = create_session(client)

    list_resp = client.get("/sessions")
    assert list_resp.status_code == 200
    ids = [s["id"] for s in list_resp.json()]
    assert session_id in ids

    messages_resp = client.get(f"/sessions/{session_id}/messages")
    assert messages_resp.status_code == 200
    assert messages_resp.json() == []

    delete_resp = client.delete(f"/sessions/{session_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json() == {"success": True}

    ids_after = [s["id"] for s in client.get("/sessions").json()]
    assert session_id not in ids_after


def test_delete_unknown_session_404(client):
    resp = client.delete("/sessions/does-not-exist")
    assert resp.status_code == 404


def test_ask_without_session_id_succeeds(client):
    # Regression test: /ask used to throw UnboundLocalError when session_id
    # was omitted, because generate_answer() was nested inside the
    # `if req.session_id:` branch instead of always running.
    resp = client.post("/ask", json={"question": "What time does the library open?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "MOCKED_ANSWER"
    assert isinstance(body["sources"], list)


def test_ask_with_unknown_session_id_404s_without_calling_llm(client, mock_call_llm):
    resp = client.post("/ask", json={"question": "test", "session_id": "does-not-exist"})
    assert resp.status_code == 404
    mock_call_llm.assert_not_called()


def test_ask_with_session_id_persists_messages_and_autotitles(client):
    session_id = create_session(client)

    short_question = "Library hours?"
    resp = client.post("/ask", json={"question": short_question, "session_id": session_id})
    assert resp.status_code == 200

    sessions = {s["id"]: s for s in client.get("/sessions").json()}
    assert sessions[session_id]["title"] == short_question

    messages = client.get(f"/sessions/{session_id}/messages").json()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == short_question
    assert messages[1]["content"] == "MOCKED_ANSWER"

    # A second question in the same session must not re-title it.
    long_question = "This is a much longer follow-up question that exceeds thirty-five characters easily."
    client.post("/ask", json={"question": long_question, "session_id": session_id})

    sessions_after = {s["id"]: s for s in client.get("/sessions").json()}
    assert sessions_after[session_id]["title"] == short_question

    messages_after = client.get(f"/sessions/{session_id}/messages").json()
    assert len(messages_after) == 4
    assert [m["role"] for m in messages_after] == ["user", "assistant", "user", "assistant"]
    assert messages_after[2]["content"] == long_question


def test_delete_session_cascades_to_messages(client):
    session_id = create_session(client)
    client.post("/ask", json={"question": "test question", "session_id": session_id})

    assert len(client.get(f"/sessions/{session_id}/messages").json()) == 2

    client.delete(f"/sessions/{session_id}")

    # Documents current behavior: no existence check on this route, so a
    # deleted (or otherwise unknown) session_id returns 200 + [] rather
    # than 404.
    resp = client.get(f"/sessions/{session_id}/messages")
    assert resp.status_code == 200
    assert resp.json() == []


def test_messages_for_unknown_session_returns_empty_list_not_404(client):
    # Documents current behavior (not a regression test for desired
    # behavior): GET /sessions/{id}/messages has no existence check.
    resp = client.get("/sessions/never-existed/messages")
    assert resp.status_code == 200
    assert resp.json() == []
