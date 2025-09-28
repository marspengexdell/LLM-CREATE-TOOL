from __future__ import annotations

import asyncio

import pytest


@pytest.mark.anyio
async def test_training_lifecycle(async_client):
    start_response = await async_client.post("/train/start", json={"steps": 5, "description": "unit-test"})
    assert start_response.status_code == 200
    payload = start_response.json()
    run_id = payload.get("runId")
    assert isinstance(run_id, str) and run_id

    status_response = await async_client.get(f"/train/status?runId={run_id}")
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["runId"] == run_id
    assert status_payload["state"] in {"pending", "running", "completed"}

    abort_response = await async_client.post("/train/abort", json={"runId": run_id})
    assert abort_response.status_code == 200

    final_state = None
    for _ in range(30):
        poll_response = await async_client.get(f"/train/status?runId={run_id}")
        assert poll_response.status_code == 200
        poll_payload = poll_response.json()
        final_state = poll_payload["state"]
        if final_state == "aborted":
            break
        await asyncio.sleep(0.1)
    else:  # pragma: no cover - would indicate the abort didn't take effect in time
        pytest.fail("Training run did not enter the aborted state in time")

    assert final_state == "aborted"


@pytest.mark.anyio
async def test_training_status_unknown_run(async_client):
    response = await async_client.get("/train/status?runId=missing")
    assert response.status_code == 404
    payload = response.json()
    assert payload["error"]["details"]["errorCode"] == "TRAINING_RUN_NOT_FOUND"


@pytest.mark.anyio
async def test_training_abort_unknown_run(async_client):
    response = await async_client.post("/train/abort", json={"runId": "missing"})
    assert response.status_code == 404
    payload = response.json()
    assert payload["error"]["details"]["errorCode"] == "TRAINING_RUN_NOT_FOUND"
