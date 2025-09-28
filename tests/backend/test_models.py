from __future__ import annotations

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
