import asyncio

import pytest


@pytest.fixture
def training_manager(workflow_main, monkeypatch):
    """Provide a fresh training job manager for each test."""

    manager = workflow_main.TrainingJobManager()
    monkeypatch.setattr(workflow_main, "TRAINING_MANAGER", manager)
    return manager


async def _wait_for_state(async_client, run_id: str, expected_states: set[str], timeout: float = 10.0):
    """Poll the legacy status endpoint until one of the expected states is observed."""

    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    last_payload = None
    while loop.time() < deadline:
        response = await async_client.get(f"/train/status?runId={run_id}")
        assert response.status_code == 200
        last_payload = response.json()
        if last_payload.get("state") in expected_states:
            return last_payload
        await asyncio.sleep(0.2)
    pytest.fail(f"Training job {run_id} did not reach states {expected_states} within timeout")


@pytest.mark.anyio
async def test_legacy_train_start_and_status(async_client, training_manager):
    response = await async_client.post(
        "/train/start",
        json={"steps": 2, "description": "integration test"},
    )
    assert response.status_code == 200
    payload = response.json()
    run_id = payload["runId"]
    assert run_id

    status_payload = await _wait_for_state(async_client, run_id, {"completed", "failed"})
    assert status_payload["runId"] == run_id
    assert status_payload["state"] == "completed"
    assert status_payload["progress"] == pytest.approx(100.0)
    assert isinstance(status_payload["metrics"], dict)


@pytest.mark.anyio
async def test_legacy_train_status_not_found(async_client, training_manager):
    response = await async_client.get("/train/status?runId=missing")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == 404
    assert error["details"]["errorCode"] == "TRAINING_RUN_NOT_FOUND"


@pytest.mark.anyio
async def test_legacy_train_abort(async_client, training_manager):
    response = await async_client.post("/train/start", json={"steps": 20})
    assert response.status_code == 200
    run_id = response.json()["runId"]

    abort_response = await async_client.post("/train/abort", json={"runId": run_id})
    assert abort_response.status_code == 200
    abort_payload = abort_response.json()
    assert abort_payload["runId"] == run_id

    status_payload = await _wait_for_state(async_client, run_id, {"aborted", "completed"})
    assert status_payload["runId"] == run_id
    assert status_payload["state"] in {"aborted", "completed"}


@pytest.mark.anyio
async def test_legacy_train_abort_not_found(async_client, training_manager):
    response = await async_client.post("/train/abort", json={"runId": "unknown"})
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == 404
    assert error["details"]["errorCode"] == "TRAINING_RUN_NOT_FOUND"
