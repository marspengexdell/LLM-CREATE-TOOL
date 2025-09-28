from __future__ import annotations

import json

import pytest


@pytest.mark.anyio("asyncio")
async def test_list_datasets_initially_empty(async_client):
    response = await async_client.get("/api/v1/datasets")

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.anyio("asyncio")
async def test_upload_dataset_persists_metadata(async_client, workflow_main, storage_paths):
    payload = {"file": ("notes.txt", b"sample data", "text/plain")}

    upload_response = await async_client.post("/api/v1/datasets/upload", files=payload)

    assert upload_response.status_code == 200
    metadata = upload_response.json()
    dataset_id = metadata["id"]

    dataset_folder = storage_paths["datasets"] / dataset_id
    stored_file = dataset_folder / "notes.txt"
    assert stored_file.exists()
    assert stored_file.read_bytes() == b"sample data"

    index_path = workflow_main.DATASETS_INDEX_PATH
    assert index_path.exists()
    index_contents = json.loads(index_path.read_text("utf-8"))
    assert any(entry["id"] == dataset_id for entry in index_contents)

    list_response = await async_client.get("/api/v1/datasets")
    assert list_response.status_code == 200
    datasets = list_response.json()
    assert len(datasets) == 1
    dataset_entry = datasets[0]
    assert dataset_entry["id"] == dataset_id
    assert dataset_entry["datasetId"] == dataset_id
    assert dataset_entry["name"] == "notes.txt"
    assert dataset_entry["storedFilename"] == "notes.txt"
    assert dataset_entry["size"] == metadata["size"]
    assert dataset_entry["mimeType"] == metadata["mimeType"]
    assert dataset_entry["type"] == metadata["type"]
    assert dataset_entry["preview"] == "sample data"


@pytest.mark.anyio("asyncio")
async def test_upload_dataset_rejects_unsupported_extension(async_client):
    payload = {"file": ("payload.exe", b"stub", "application/octet-stream")}

    response = await async_client.post("/api/v1/datasets/upload", files=payload)

    assert response.status_code == 400
    body = response.json()
    error = body.get("detail") or body.get("error") or {}
    details = error.get("details", {})
    error_code = error.get("error_code") or details.get("errorCode")
    assert error_code == "DATASET_UNSUPPORTED_TYPE"
    assert ".exe" not in details.get("allowedExtensions", [])
