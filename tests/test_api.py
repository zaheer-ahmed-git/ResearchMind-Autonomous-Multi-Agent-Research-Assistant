"""FastAPI integration tests."""

from fastapi.testclient import TestClient

from research_agent.api.deps import clear_runs
from research_agent.api.main import create_app


def setup_function() -> None:
    clear_runs()


def test_health_endpoint():
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_research_run_and_traces():
    client = TestClient(create_app())
    created = client.post(
        "/research/run",
        json={"query": "How does adaptive ranking reduce hallucinations?"},
    )
    assert created.status_code == 200
    body = created.json()
    run_id = body["run_id"]
    assert body["report"]
    assert body["traces"]

    listed = client.get("/research/runs")
    assert listed.status_code == 200
    assert any(item["run_id"] == run_id for item in listed.json())

    traces = client.get(f"/research/runs/{run_id}/traces")
    assert traces.status_code == 200
    assert traces.json()["run_id"] == run_id
    assert len(traces.json()["traces"]) >= 2
