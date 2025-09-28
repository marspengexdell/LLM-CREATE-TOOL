from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest


@pytest.mark.anyio("asyncio")
async def test_model_registry_crud(async_client):
    initial = await async_client.get("/api/v1/models")
    assert initial.status_code == 200
    assert initial.json() == []

    payload = {
        "name": "Quantized Gemini",
        "source": "runs/demo/model.gguf",
        "quantization": "int4",
        "description": "Demo entry",
        "metadata": {"artifactPath": "runs/demo/model.gguf"},
    }

    create_response = await async_client.post("/api/v1/models", json=payload)
    assert create_response.status_code == 201
    created = create_response.json()
    assert created["name"] == payload["name"]
    assert created["source"] == payload["source"]
    assert created["quantization"] == payload["quantization"]
    assert created["description"] == payload["description"]
    assert "registeredAt" in created
    assert created["metadata"]["artifactPath"] == payload["metadata"]["artifactPath"]

    listing = await async_client.get("/api/v1/models")
    assert listing.status_code == 200
    models = listing.json()
    assert len(models) == 1
    assert models[0]["id"] == created["id"]

    delete_response = await async_client.request("DELETE", f"/api/v1/models/{created['id']}")
    assert delete_response.status_code == 204

    after_delete = await async_client.get("/api/v1/models")
    assert after_delete.status_code == 200
    assert after_delete.json() == []


def test_concurrent_model_registry_updates(workflow_main, storage_paths):
    total_entries = 24

    def register_model(index: int) -> str:
        entry = {
            "name": f"Model {index}",
            "source": f"runs/model-{index}.bin",
            "description": f"Synthetic entry {index}",
        }
        return workflow_main._register_model_entry(entry)["id"]

    with ThreadPoolExecutor(max_workers=8) as executor:
        ids = [future.result() for future in (executor.submit(register_model, i) for i in range(total_entries))]

    content = workflow_main.MODELS_INDEX_PATH.read_text("utf-8")
    registry = json.loads(content) if content else []
    assert len(registry) == total_entries
    assert {item["id"] for item in registry} == set(ids)

    def reregister_model(model_id: str) -> None:
        workflow_main._register_model_entry(
            {
                "id": model_id,
                "name": f"Updated {model_id}",
                "source": f"runs/{model_id}.bin",
                "metadata": {"updated": True},
            }
        )

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(reregister_model, ids))

    updated_registry = workflow_main._model_registry_entries()
    assert len(updated_registry) == total_entries
    assert {item["id"] for item in updated_registry} == set(ids)
    assert all(item["metadata"].get("updated") for item in updated_registry)

    removals = ids[: total_entries // 3]

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(workflow_main._remove_model_entry, removals))

    assert all(results)

    remaining = workflow_main._model_registry_entries()
    assert len(remaining) == total_entries - len(removals)
    assert {item["id"] for item in remaining} == set(ids) - set(removals)

    # Ensure the registry file remains valid JSON after concurrent mutations.
    json.loads(workflow_main.MODELS_INDEX_PATH.read_text("utf-8"))
