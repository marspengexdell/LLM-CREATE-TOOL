from __future__ import annotations

import asyncio

import pytest


@pytest.mark.anyio
async def test_training_lifecycle(async_client):
    request_payload = {
        "datasetId": "unit-dataset",
        "modelId": "unit-model",
        "hyperparameters": {"learningRate": 0.1},
        "notes": "integration-test",
    }
    start_response = await async_client.post("/api/v1/train/start", json=request_payload)
    assert start_response.status_code == 200
    payload = start_response.json()
    job_id = payload.get("jobId")
    assert isinstance(job_id, str) and job_id
    assert payload["datasetId"] == request_payload["datasetId"]
    assert payload["modelId"] == request_payload["modelId"]
    assert payload["status"] in {"running", "queued"}

    status_response = await async_client.get(f"/api/v1/train/status?jobId={job_id}")
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["jobId"] == job_id
    assert status_payload["status"] in {"running", "completed"}
    assert 0.0 <= status_payload["progress"] <= 1.0

    abort_response = await async_client.post("/api/v1/train/abort", json={"jobId": job_id})
    assert abort_response.status_code == 200

    final_state = None
    for _ in range(30):
        poll_response = await async_client.get(f"/api/v1/train/status?jobId={job_id}")
        assert poll_response.status_code == 200
        poll_payload = poll_response.json()
        final_state = poll_payload["status"]
        if final_state == "aborted":
            break
        await asyncio.sleep(0.1)
    else:  # pragma: no cover - would indicate the abort didn't take effect in time
        pytest.fail("Training job did not enter the aborted state in time")

    assert final_state == "aborted"


@pytest.mark.anyio
async def test_training_status_unknown_job(async_client):
    response = await async_client.get("/api/v1/train/status?jobId=missing")
    assert response.status_code == 404
    payload = response.json()
    assert payload["error"]["details"]["errorCode"] == "TRAINING_JOB_NOT_FOUND"


@pytest.mark.anyio
async def test_training_abort_unknown_job(async_client):
    response = await async_client.post("/api/v1/train/abort", json={"jobId": "missing"})
    assert response.status_code == 404
    payload = response.json()
    assert payload["error"]["details"]["errorCode"] == "TRAINING_JOB_NOT_FOUND"
