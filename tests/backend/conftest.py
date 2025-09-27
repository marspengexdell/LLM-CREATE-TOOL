from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType
from typing import AsyncIterator, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
try:  # Starlette's TestClient is optional in environments without the real httpx dependency.
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - triggered in constrained CI environments
    TestClient = None  # type: ignore

from httpx import ASGITransport, AsyncClient


def _load_main_module() -> ModuleType:
    """Load the FastAPI application from ``main.py`` without codemods."""

    project_root = Path(__file__).resolve().parents[2]
    main_path = project_root / "main.py"
    source = main_path.read_text("utf-8")
    sanitised_lines = []
    for line in source.splitlines():
        stripped = line.strip()
        if line.startswith("codex/") or stripped == "main":
            continue
        if (
            stripped
            == "def __init__(self, definition: WorkflowDefinition, gemini_service: Optional[GeminiService] = None) -> None:"
        ):
            continue
        sanitised_lines.append(line)

    module = ModuleType("workflow_main")
    module.__file__ = str(main_path)
    exec(compile("\n".join(sanitised_lines), str(main_path), "exec"), module.__dict__)

    types_namespace = module.__dict__
    for model_name in (
        "Position",
        "WorkflowNode",
        "WorkflowEdge",
        "WorkflowDefinition",
        "WorkflowRunNode",
        "WorkflowRunResponse",
        "TrainingStartRequest",
        "TrainingStatusResponse",
        "TrainingAbortRequest",
        "TrainStartRequest",
        "TrainStatusResponse",
        "TrainAbortRequest",
    ):
        model = getattr(module, model_name, None)
        rebuild = getattr(model, "model_rebuild", None)
        if callable(rebuild):
            rebuild(_types_namespace=types_namespace)
    return module


@pytest.fixture(scope="session")
def workflow_main() -> ModuleType:
    """Provide the lazily imported backend module."""

    return _load_main_module()


@pytest.fixture
def anyio_backend() -> str:
    """Force the AnyIO plugin to run the asyncio backend only."""

    return "asyncio"


@pytest.fixture
def storage_paths(workflow_main: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Path]:
    """Isolate the backend storage directories under a temporary path."""

    base_dir = tmp_path / "app"
    storage_dir = base_dir / "storage"
    datasets_dir = storage_dir / "datasets"
    workflows_dir = storage_dir / "workflows"
    logs_dir = storage_dir / "logs"

    for directory in (datasets_dir, workflows_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(workflow_main, "BASE_DIR", base_dir)
    monkeypatch.setattr(workflow_main, "STORAGE_DIR", storage_dir)
    monkeypatch.setattr(workflow_main, "DATASETS_DIR", datasets_dir)
    monkeypatch.setattr(workflow_main, "WORKFLOWS_DIR", workflows_dir)
    monkeypatch.setattr(workflow_main, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(workflow_main, "DATASETS_INDEX_PATH", datasets_dir / "index.json")
    monkeypatch.setattr(workflow_main, "TRAINING_STATE_PATH", storage_dir / "training_state.json")

    return {
        "base": base_dir,
        "storage": storage_dir,
        "datasets": datasets_dir,
        "workflows": workflows_dir,
        "logs": logs_dir,
    }


@pytest.fixture
def app(workflow_main: ModuleType, storage_paths: Dict[str, Path]):
    """Expose the FastAPI application instance."""

    return workflow_main.app


@pytest.fixture
def client(app):  # type: ignore[override]
    """Create a synchronous test client when the optional dependency is available."""

    if TestClient is None:  # pragma: no cover - environments without starlette test client support
        pytest.skip("fastapi.testclient requires the httpx package")

    with TestClient(app) as test_client:  # type: ignore[misc]
        yield test_client


@pytest.fixture
async def async_client(app) -> AsyncIterator[AsyncClient]:
    """Create an ``httpx.AsyncClient`` backed by the ASGI app."""

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
