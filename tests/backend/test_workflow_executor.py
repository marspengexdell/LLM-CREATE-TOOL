from __future__ import annotations

from copy import deepcopy

import pytest


def _load_executor_types(workflow_main):
    """Helper to access the dynamic workflow module classes."""

    return {
        "Position": workflow_main.Position,
        "WorkflowNode": workflow_main.WorkflowNode,
        "WorkflowEdge": workflow_main.WorkflowEdge,
        "WorkflowDefinition": workflow_main.WorkflowDefinition,
        "WorkflowExecutor": workflow_main.WorkflowExecutor,
        "GEMINI_DEFAULT_MODEL": workflow_main.GEMINI_DEFAULT_MODEL,
    }


def _basic_workflow_definition():
    return {
        "name": "Example workflow",
        "description": "single node",
        "nodes": [
            {
                "id": "text-source",
                "type": "text_input",
                "position": {"x": 0, "y": 0},
                "data": {"text": "hello"},
            }
        ],
        "edges": [],
    }


@pytest.mark.anyio("asyncio")
async def test_workflows_round_trip(async_client, workflow_main, storage_paths):
    definition = _basic_workflow_definition()

    save_response = await async_client.post("/api/v1/workflows/save", json=definition)
    assert save_response.status_code == 200
    workflow_id = save_response.json()["id"]

    workflows = await async_client.get("/api/v1/workflows")
    assert workflows.status_code == 200
    listing = workflows.json()
    assert any(item["id"] == workflow_id for item in listing)

    detail = await async_client.get(f"/api/v1/workflows/{workflow_id}")
    assert detail.status_code == 200
    stored_definition = detail.json()
    assert stored_definition["id"] == workflow_id
    assert stored_definition["nodes"][0]["id"] == "text-source"

    workflow_path = storage_paths["workflows"] / f"{workflow_id}.json"
    assert workflow_path.exists()


@pytest.mark.anyio("asyncio")
async def test_run_workflow_success(async_client):
    definition = _basic_workflow_definition()

    run_response = await async_client.post("/api/v1/workflow/run", json=definition)
    assert run_response.status_code == 200
    payload = run_response.json()
    assert "run_id" in payload
    assert len(payload["nodes"]) == 1
    node_state = payload["nodes"][0]
    assert node_state["status"] == "done"
    assert node_state["data"]["text"] == "hello"


@pytest.mark.anyio("asyncio")
async def test_run_workflow_requires_nodes(async_client):
    response = await async_client.post("/api/v1/workflow/run", json={"nodes": [], "edges": []})
    assert response.status_code == 400
    body = response.json()
    error = body.get("detail") or body.get("error") or {}
    details = error.get("details", {})
    error_code = (
        error.get("error_code")
        or error.get("errorCode")
        or details.get("errorCode")
        or error.get("code")
    )
    assert error_code == "WORKFLOW_EMPTY"


@pytest.mark.anyio("asyncio")
async def test_run_workflow_reports_node_failure(async_client):
    definition = _basic_workflow_definition()
    failing_node = {
        "id": "report",
        "type": "generate_report",
        "position": {"x": 400, "y": 0},
        "data": {"prompt": "Summarise"},
    }
    edge = {
        "id": "link",
        "fromNode": "text-source",
        "fromPort": "out",
        "toNode": "report",
        "toPort": "in",
    }

    definition_with_report = deepcopy(definition)
    definition_with_report["nodes"].append(failing_node)
    definition_with_report["edges"].append(edge)

    run_response = await async_client.post("/api/v1/workflow/run", json=definition_with_report)
    assert run_response.status_code == 200
    payload = run_response.json()
    statuses = {node["id"]: node["status"] for node in payload["nodes"]}
    assert statuses["text-source"] == "done"
    assert statuses["report"] == "done"
    report_node = next(node for node in payload["nodes"] if node["id"] == "report")
    report_data = report_node.get("data", {})
    assert "[Gemini placeholder]" in report_data.get("report", "")


@pytest.mark.anyio("asyncio")
async def test_registry_publish_registers_model(async_client):
    workflow = {
        "name": "Registry publish",
        "description": "register quantized model",
        "nodes": [
            {
                "id": "quantize",
                "type": "quantize_model",
                "position": {"x": 0, "y": 0},
                "data": {"bits": 4, "scheme": "nf4"},
            },
            {
                "id": "publish",
                "type": "registry_publish",
                "position": {"x": 200, "y": 0},
                "data": {"modelName": "Quant Demo", "description": "From workflow"},
            },
        ],
        "edges": [
            {
                "id": "link",
                "fromNode": "quantize",
                "fromPort": "model",
                "toNode": "publish",
                "toPort": "model",
            }
        ],
    }

    run_response = await async_client.post("/api/v1/workflow/run", json=workflow)
    assert run_response.status_code == 200
    payload = run_response.json()
    publish_state = next(node for node in payload["nodes"] if node["id"] == "publish")
    registry_entry = publish_state["data"].get("registryEntry")
    assert registry_entry is not None
    assert registry_entry["name"] == "Quant Demo"
    assert registry_entry["quantization"] == "int4"

    models_response = await async_client.get("/api/v1/models")
    assert models_response.status_code == 200
    models = models_response.json()
    assert any(model["id"] == registry_entry["id"] for model in models)
def test_generate_report_placeholder_without_gemini(workflow_main):
    types = _load_executor_types(workflow_main)
    Position = types["Position"]
    WorkflowNode = types["WorkflowNode"]
    WorkflowEdge = types["WorkflowEdge"]
    WorkflowDefinition = types["WorkflowDefinition"]
    WorkflowExecutor = types["WorkflowExecutor"]
    default_model = types["GEMINI_DEFAULT_MODEL"]

    text_node = WorkflowNode(
        id="text-source",
        type="text_input",
        position=Position(x=0, y=0),
        data={"text": "hello"},
    )
    report_node = WorkflowNode(
        id="report",
        type="generate_report",
        position=Position(x=400, y=0),
        data={"prompt": "Summarise"},
    )
    edge = WorkflowEdge(
        id="link",
        fromNode="text-source",
        fromPort="out",
        toNode="report",
        toPort="in",
    )

    definition = WorkflowDefinition(nodes=[text_node, report_node], edges=[edge])

    executor = WorkflowExecutor(definition, gemini_service=None)
    response = executor.run()

    report_state = next(node for node in response.nodes if node.id == "report")
    assert report_state.status == "done"

    report_text = report_state.data["report"]
    assert "[Gemini placeholder]" in report_text
    assert "hello" in report_state.data["aggregated_inputs"]
    assert report_state.data["model"] == default_model
