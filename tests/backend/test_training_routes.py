from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


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


@pytest.mark.anyio
async def test_training_state_recovers_from_truncated_file(async_client, storage_paths, workflow_main):
    request_payload = {
        "datasetId": "dataset-backup",
        "modelId": "model-backup",
        "hyperparameters": {"learningRate": 0.5},
    }

    start_response = await async_client.post("/api/v1/train/start", json=request_payload)
    assert start_response.status_code == 200
    job_id = start_response.json()["jobId"]

    status_response = await async_client.get(f"/api/v1/train/status?jobId={job_id}")
    assert status_response.status_code == 200

    state_path = storage_paths["storage"] / "training_state.json"
    backup_path = state_path.with_suffix(state_path.suffix + ".bak")

    assert backup_path.exists(), "Expected a backup file after persisting training state"
    backup_state = _read_json(backup_path)
    assert job_id in backup_state.get("jobs", {})

    state_path.write_text("{", encoding="utf-8")

    workflow_main.TRAINING_CONTROLLER = workflow_main.TrainingController(
        workflow_main.TRAINING_STATE_PATH, workflow_main.LOGGER
    )

    recovered_response = await async_client.get(f"/api/v1/train/status?jobId={job_id}")
    assert recovered_response.status_code == 200
    recovered_payload = recovered_response.json()
    assert recovered_payload["jobId"] == job_id
    assert recovered_payload["datasetId"] == request_payload["datasetId"]
