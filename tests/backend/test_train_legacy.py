import asyncio

import pytest


async def _wait_for_status(
    async_client,
    job_id: str,
    expected_states: set[str],
    *,
    timeout: float = 10.0,
    use_legacy_param: bool = False,
):
    """Poll the status endpoint until a matching state is observed."""

    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    last_payload = None
    query_param = "runId" if use_legacy_param else "jobId"

    while loop.time() < deadline:
        response = await async_client.get(f"/train/status?{query_param}={job_id}")
        assert response.status_code == 200
        last_payload = response.json()
        if last_payload.get("status") in expected_states:
            return last_payload
        await asyncio.sleep(0.2)

    pytest.fail(f"Training job {job_id} did not reach states {expected_states} within timeout")


@pytest.mark.anyio
async def test_legacy_routes_alias_new_controller(async_client):
    payload = {
        "datasetId": "legacy-dataset",
        "modelId": "legacy-model",
        "hyperparameters": {"epochs": 1},
    }

    start_response = await async_client.post("/train/start", json=payload)
    assert start_response.status_code == 200
    start_payload = start_response.json()
    job_id = start_payload["jobId"]
    assert start_payload["status"] in {"running", "queued"}

    api_status = await async_client.get(f"/api/v1/train/status?jobId={job_id}")
    legacy_status = await async_client.get(f"/train/status?runId={job_id}")
    assert api_status.status_code == legacy_status.status_code == 200
    api_payload = api_status.json()
    legacy_payload = legacy_status.json()
    assert api_payload["jobId"] == legacy_payload["jobId"] == job_id
    assert api_payload["datasetId"] == legacy_payload["datasetId"]
    assert api_payload["modelId"] == legacy_payload["modelId"]
    assert api_payload["status"] == legacy_payload["status"]

    abort_response = await async_client.post("/train/abort", json={"runId": job_id})
    assert abort_response.status_code == 200

    final_status = await _wait_for_status(async_client, job_id, {"aborted"}, use_legacy_param=True)
    assert final_status["status"] == "aborted"


@pytest.mark.anyio
async def test_legacy_status_not_found(async_client):
    response = await async_client.get("/train/status?runId=unknown")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["details"]["errorCode"] == "TRAINING_JOB_NOT_FOUND"


@pytest.mark.anyio
async def test_legacy_abort_not_found(async_client):
    response = await async_client.post("/train/abort", json={"runId": "unknown"})
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["details"]["errorCode"] == "TRAINING_JOB_NOT_FOUND"
