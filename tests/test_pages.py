def test_index_page_loads_and_links_static_assets(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "/static/styles.css" in resp.text
    assert "/static/script.js" in resp.text


def test_manage_page_loads(client):
    resp = client.get("/manage")
    assert resp.status_code == 200


def test_static_assets_are_served(client):
    css_resp = client.get("/static/styles.css")
    assert css_resp.status_code == 200
    assert len(css_resp.text) > 0

    js_resp = client.get("/static/script.js")
    assert js_resp.status_code == 200
    assert len(js_resp.text) > 0
