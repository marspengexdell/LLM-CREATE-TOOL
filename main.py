import base64
import json
import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Environment & Configuration
# ---------------------------------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
STORAGE_DIR = BASE_DIR / "storage"
DATASETS_DIR = STORAGE_DIR / "datasets"
WORKFLOWS_DIR = STORAGE_DIR / "workflows"
LOGS_DIR = STORAGE_DIR / "logs"

for directory in (DATASETS_DIR, WORKFLOWS_DIR, LOGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DATASETS_INDEX_PATH = DATASETS_DIR / "index.json"
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024  # 20MB
PREVIEW_LIMIT_BYTES = 1 * 1024 * 1024  # 1MB for inline previews

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".csv"}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOGS_DIR.mkdir(exist_ok=True)
LOGGER = logging.getLogger("workflow-engine")
LOGGER.setLevel(logging.INFO)

if not LOGGER.handlers:
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOGS_DIR / "backend.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(file_handler)

# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(title="Workflow Orchestrator API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class Position(BaseModel):
    x: float
    y: float


class WorkflowNodePayload(BaseModel):
    id: str
    type: str
    position: Position
    data: Dict[str, Any] = Field(default_factory=dict)


class WorkflowEdgePayload(BaseModel):
    id: str
    fromNode: str
    fromPort: str
    toNode: str
    toPort: str


class WorkflowDefinition(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    nodes: List[WorkflowNodePayload]
    edges: List[WorkflowEdgePayload]


class WorkflowRunResponseNode(BaseModel):
    id: str
    type: str
    x: float
    y: float
    data: Dict[str, Any]
    status: str


class WorkflowRunResponse(BaseModel):
    nodes: List[WorkflowRunResponseNode]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        LOGGER.warning("Failed to decode JSON at %s. Returning default.", path)
        return default


def _save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _dataset_index() -> List[Dict[str, Any]]:
    return _load_json(DATASETS_INDEX_PATH, [])


def _write_dataset_index(entries: List[Dict[str, Any]]) -> None:
    _save_json(DATASETS_INDEX_PATH, entries)


def _detect_dataset_type(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    if extension in IMAGE_EXTENSIONS:
        return "image"
    if extension in TEXT_EXTENSIONS:
        return "text"
    return "file"


def _detect_mime_type(extension: str) -> str:
    mapping = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".json": "application/json",
        ".csv": "text/csv",
    }
    return mapping.get(extension.lower(), "application/octet-stream")


def _store_uploaded_file(dataset_id: str, upload: UploadFile, content: bytes) -> Path:
    extension = Path(upload.filename or "").suffix
    stored_name = f"{dataset_id}{extension}"
    stored_path = DATASETS_DIR / stored_name
    with stored_path.open("wb") as f:
        f.write(content)
    return stored_path


def _prepare_dataset_metadata(dataset_id: str, upload: UploadFile, stored_path: Path, size_bytes: int, preview: Optional[str]) -> Dict[str, Any]:
    extension = stored_path.suffix.lower()
    dataset_type = _detect_dataset_type(upload.filename or stored_path.name)
    return {
        "id": dataset_id,
        "name": upload.filename or stored_path.name,
        "filename": stored_path.name,
        "type": dataset_type,
        "size": size_bytes,
        "mimeType": _detect_mime_type(extension),
        "preview": preview,
        "uploadedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _load_workflow_file(workflow_id: str) -> Dict[str, Any]:
    workflow_path = WORKFLOWS_DIR / f"{workflow_id}.json"
    if not workflow_path.exists():
        raise HTTPException(status_code=404, detail="Workflow not found")
    with workflow_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _gemini_generate(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "Gemini API key not configured. Returning placeholder report."

    try:
        model = genai.GenerativeModel(DEFAULT_GEMINI_MODEL)
        response = model.generate_content(prompt)
        if hasattr(response, "text") and response.text:
            return response.text
        if response.candidates:
            # Fall back to the first candidate's text if available.
            return response.candidates[0].content.parts[0].text  # type: ignore[index]
        return "Gemini response did not include text content."
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Gemini generation failed: %s", exc)
        return "Failed to generate report with Gemini. Placeholder response returned."


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/models")
def list_models() -> List[Dict[str, Any]]:
    """Return a static list of available models."""
    models = [
        {
            "id": "gemini-1.5-flash",
            "name": "Gemini 1.5 Flash",
            "description": "Fast multimodal model suitable for real-time interactions.",
        },
        {
            "id": "gemini-1.5-pro",
            "name": "Gemini 1.5 Pro",
            "description": "Higher quality Gemini model for complex reasoning tasks.",
        },
    ]
    return models


@app.get("/api/v1/datasets")
def list_datasets() -> List[Dict[str, Any]]:
    """Return dataset metadata stored on disk."""
    entries = _dataset_index()
    return entries


@app.post("/api/v1/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename")

    content = await file.read()
    size_bytes = len(content)

    if size_bytes == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if size_bytes > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds maximum allowed size of 20MB")

    dataset_id = str(uuid4())
    stored_path = _store_uploaded_file(dataset_id, file, content)

    preview: Optional[str] = None
    extension = stored_path.suffix.lower()
    if extension in IMAGE_EXTENSIONS and size_bytes <= PREVIEW_LIMIT_BYTES:
        mime_type = _detect_mime_type(extension)
        encoded = base64.b64encode(content).decode("utf-8")
        preview = f"data:{mime_type};base64,{encoded}"

    metadata = _prepare_dataset_metadata(dataset_id, file, stored_path, size_bytes, preview)

    entries = _dataset_index()
    entries = [entry for entry in entries if entry.get("id") != dataset_id]
    entries.append(metadata)
    _write_dataset_index(entries)

    LOGGER.info("Uploaded dataset %s (%s)", metadata["name"], dataset_id)
    return metadata


@app.get("/api/v1/workflows")
def list_workflows() -> List[Dict[str, Any]]:
    workflows: List[Dict[str, Any]] = []
    for workflow_path in sorted(WORKFLOWS_DIR.glob("*.json")):
        try:
            data = _load_json(workflow_path, {})
            workflow_id = workflow_path.stem
            workflows.append(
                {
                    "id": workflow_id,
                    "name": data.get("name") or f"Workflow {workflow_id[:8]}",
                    "description": data.get("description", ""),
                    "updatedAt": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(workflow_path.stat().st_mtime)
                    ),
                }
            )
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Failed to load workflow metadata from %s: %s", workflow_path, exc)
    return workflows


@app.get("/api/v1/workflows/{workflow_id}")
def get_workflow(workflow_id: str) -> Dict[str, Any]:
    return _load_workflow_file(workflow_id)


@app.post("/api/v1/workflows/save")
def save_workflow(definition: WorkflowDefinition) -> Dict[str, Any]:
    workflow_id = definition.id or str(uuid4())
    payload = definition.dict()
    payload["id"] = workflow_id
    payload.setdefault("name", f"Workflow {workflow_id[:8]}")

    workflow_path = WORKFLOWS_DIR / f"{workflow_id}.json"
    _save_json(workflow_path, payload)

    LOGGER.info("Saved workflow %s", workflow_id)
    return {"id": workflow_id}


# ---------------------------------------------------------------------------
# Workflow Execution
# ---------------------------------------------------------------------------


def _gather_inputs(
    node_id: str,
    incoming_edges: Dict[str, List[WorkflowEdgePayload]],
    node_outputs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    inputs: Dict[str, Any] = {}
    for edge in incoming_edges.get(node_id, []):
        source_output = node_outputs.get(edge.fromNode, {})
        if edge.fromPort in source_output:
            inputs[edge.toPort] = source_output[edge.fromPort]
    return inputs


def _execute_node(
    node: WorkflowNodePayload,
    current_state: Dict[str, Any],
    inputs: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    node_type = node.type
    data = dict(current_state)

    if node_type == "image_input":
        return {"out": data.get("image")}, data

    if node_type == "text_input":
        return {"out": data.get("text")}, data

    if node_type == "data_hub":
        dataset_id = data.get("datasetId")
        dataset_name = data.get("datasetName")
        if dataset_id:
            entries = {entry["id"]: entry for entry in _dataset_index()}
            selected = entries.get(dataset_id)
            if selected:
                return {"out": selected}, data
        return {"out": {"datasetId": dataset_id, "datasetName": dataset_name}}, data

    if node_type == "model_hub":
        model_id = data.get("modelId")
        return {"out": {"modelId": model_id}}, data

    if node_type == "image_classifier":
        image = inputs.get("image")
        model = inputs.get("model")
        classification = {
            "label": "unknown",
            "confidence": 0.0,
            "model": model,
            "notes": "Placeholder classification result.",
            "received": bool(image),
        }
        return {"out": classification}, data

    if node_type == "decision_logic":
        condition = data.get("condition") or "bool(input)"
        input_value = inputs.get("in")
        context = {"input": input_value}
        try:
            decision = bool(eval(condition, {"__builtins__": {}}, context))  # noqa: S307
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Decision node %s failed to evaluate condition '%s': %s", node.id, condition, exc)
            decision = bool(input_value)
        data["decision"] = decision
        return {"true": input_value if decision else None, "false": None if decision else input_value}, data

    if node_type == "generate_report":
        prompt_template = data.get("prompt") or "Generate a concise report based on the provided data."
        upstream = inputs.get("in")
        context_snippet = json.dumps(upstream, ensure_ascii=False, indent=2) if upstream is not None else "No upstream data provided."
        prompt = f"{prompt_template}\n\nContext:\n{context_snippet}"
        report = _gemini_generate(prompt)
        data["report"] = report
        return {"out": report}, data

    if node_type in {"image_output", "text_output"}:  # Future extension
        return {"out": inputs}, data

    # Default passthrough
    return inputs, data


@app.post("/api/v1/workflow/run", response_model=WorkflowRunResponse)
def run_workflow(definition: WorkflowDefinition) -> WorkflowRunResponse:
    nodes_map = {node.id: node for node in definition.nodes}
    for edge in definition.edges:
        if edge.fromNode not in nodes_map or edge.toNode not in nodes_map:
            raise HTTPException(status_code=400, detail=f"Edge {edge.id} references unknown nodes")

    adjacency: Dict[str, List[str]] = defaultdict(list)
    indegree: Dict[str, int] = {node_id: 0 for node_id in nodes_map}
    incoming_edges: Dict[str, List[WorkflowEdgePayload]] = defaultdict(list)

    for edge in definition.edges:
        adjacency[edge.fromNode].append(edge.toNode)
        indegree[edge.toNode] += 1
        incoming_edges[edge.toNode].append(edge)

    queue: deque[str] = deque([node_id for node_id, degree in indegree.items() if degree == 0])
    if not queue and nodes_map:
        raise HTTPException(status_code=400, detail="Workflow has no entry nodes (cycle detected)")

    execution_order: List[str] = []
    while queue:
        current = queue.popleft()
        execution_order.append(current)
        for neighbor in adjacency[current]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    if len(execution_order) != len(nodes_map):
        raise HTTPException(status_code=400, detail="Workflow contains cycles and cannot be executed")

    node_outputs: Dict[str, Dict[str, Any]] = {}
    response_nodes: Dict[str, WorkflowRunResponseNode] = {}

    nodes_state: Dict[str, Dict[str, Any]] = {
        node.id: {
            "id": node.id,
            "type": node.type,
            "x": node.position.x,
            "y": node.position.y,
            "data": dict(node.data),
            "status": "pending",
        }
        for node in definition.nodes
    }

    for node_id in execution_order:
        node_payload = nodes_map[node_id]
        node_state = nodes_state[node_id]
        start_time = time.time()
        inputs = _gather_inputs(node_id, incoming_edges, node_outputs)
        try:
            outputs, updated_data = _execute_node(node_payload, node_state.get("data", {}), inputs)
            node_state["data"] = updated_data
            node_state["status"] = "done"
            node_outputs[node_id] = outputs
            elapsed = (time.time() - start_time) * 1000
            LOGGER.info("Node %s (%s) executed in %.2fms", node_id, node_payload.type, elapsed)
        except Exception as exc:  # pylint: disable=broad-except
            node_state["status"] = "failed"
            node_state.setdefault("data", {})["error"] = str(exc)
            node_outputs[node_id] = {}
            LOGGER.exception("Node %s (%s) failed: %s", node_id, node_payload.type, exc)

        response_nodes[node_id] = WorkflowRunResponseNode(**node_state)  # type: ignore[arg-type]

    ordered_response_nodes = [response_nodes[node.id] for node in definition.nodes]
    return WorkflowRunResponse(nodes=ordered_response_nodes)
