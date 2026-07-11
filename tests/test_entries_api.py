def test_get_entry_not_found(client):
    resp = client.get("/api/entries/does-not-exist")
    assert resp.status_code == 404


def test_update_entry_not_found(client):
    resp = client.put(
        "/api/entries/does-not-exist",
        json={"name": "X", "type": "facility"},
    )
    assert resp.status_code == 404


def test_pagination_boundaries(client):
    # 5 fixture entries seeded at startup.
    resp = client.get("/api/entries", params={"page": 1, "per_page": 2})
    data = resp.json()
    assert data["total"] == 5
    assert data["total_pages"] == 3
    assert len(data["entries"]) == 2

    # A page beyond the last page returns an empty list, not an error.
    resp = client.get("/api/entries", params={"page": 99, "per_page": 2})
    assert resp.status_code == 200
    assert resp.json()["entries"] == []

    # per_page larger than total returns everything on page 1.
    resp = client.get("/api/entries", params={"page": 1, "per_page": 100})
    assert len(resp.json()["entries"]) == 5


def test_create_read_update_delete_lifecycle(client):
    body = {
        "name": "Test Bookstore",
        "type": "facility",
        "location": "Building 9",
        "hours": {"monday_friday": "9:00 AM - 6:00 PM"},
        "description": "Sells textbooks and merchandise.",
        "contact": "bookstore@fixture.test",
        "additional_info": ["Buyback week is the last week of finals", "Price match available"],
    }

    create_resp = client.post("/api/entries", json=body)
    assert create_resp.status_code == 201
    created = create_resp.json()
    entry_id = created["id"]

    # JSON round-trip: hours dict and additional_info list survive create -> read-back.
    assert created["hours"] == body["hours"]
    assert created["additional_info"] == body["additional_info"]

    get_resp = client.get(f"/api/entries/{entry_id}")
    assert get_resp.status_code == 200
    fetched = get_resp.json()
    assert fetched["name"] == "Test Bookstore"
    assert fetched["hours"] == body["hours"]
    assert fetched["additional_info"] == body["additional_info"]

    update_body = {**body, "name": "Test Bookstore & Cafe", "description": "Now also sells coffee."}
    update_resp = client.put(f"/api/entries/{entry_id}", json=update_body)
    assert update_resp.status_code == 200
    assert update_resp.json()["name"] == "Test Bookstore & Cafe"

    delete_resp = client.delete(f"/api/entries/{entry_id}")
    assert delete_resp.status_code == 204

    assert client.get(f"/api/entries/{entry_id}").status_code == 404


def test_new_entry_is_immediately_retrievable_via_ask(client, mock_call_llm):
    body = {
        "name": "Sunset Planetarium",
        "type": "facility",
        "location": "Building 12",
        "description": "Hosts astronomy shows and telescope viewing nights for students.",
        "contact": "planetarium@fixture.test",
    }
    created = client.post("/api/entries", json=body).json()

    ask_resp = client.post("/ask", json={"question": "Where can I see a telescope viewing night?"})
    assert ask_resp.status_code == 200
    source_ids = [s["id"] for s in ask_resp.json()["sources"]]
    assert created["id"] in source_ids

    # The retrieved document content actually reached the (mocked) LLM prompt.
    prompt_sent = mock_call_llm.call_args.args[0]
    assert "Sunset Planetarium" in prompt_sent


def test_deleted_entry_is_no_longer_retrievable(client):
    body = {
        "name": "Zephyr Observatory",
        "type": "facility",
        "description": "A rooftop observatory exclusively for astronomy club members.",
    }
    created = client.post("/api/entries", json=body).json()
    entry_id = created["id"]

    ask_resp = client.post("/ask", json={"question": "Tell me about the rooftop observatory for astronomy club"})
    assert entry_id in [s["id"] for s in ask_resp.json()["sources"]]

    client.delete(f"/api/entries/{entry_id}")

    ask_resp_after = client.post("/ask", json={"question": "Tell me about the rooftop observatory for astronomy club"})
    assert entry_id not in [s["id"] for s in ask_resp_after.json()["sources"]]
