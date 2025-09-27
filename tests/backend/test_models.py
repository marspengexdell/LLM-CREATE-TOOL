from __future__ import annotations

import pytest


@pytest.mark.anyio("asyncio")
async def test_list_models(async_client):
    response = await async_client.get("/api/v1/models")

    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    assert {model["id"] for model in models} == {"gemini-1.5-flash", "gemini-1.5-pro"}
    for model in models:
        assert {"id", "name", "description"} <= set(model)
