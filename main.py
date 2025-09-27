"""FastAPI backend for the workflow builder UI.

The application intentionally keeps the implementation small so that it can
be extended quickly during prototyping.  The front-end expects a handful of
REST endpoints that provide model metadata, dataset management, workflow
persistence and a stub workflow executor.  The executor performs a basic
 topological sort before running each node with placeholder logic so that
 the UI can display meaningful status updates.
"""

from __future__ import annotations

import ast
import base64
import copy
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

codex/rewrite-backend-using-fastapi-and-implement-routes-k68a2h

try:  # Optional import to allow unit tests without the dependency installed.
    import google.generativeai as genai  # type: ignore
    from google.api_core.exceptions import GoogleAPIError  # type: ignore
except Exception:  # pragma: no cover - fallback when package missing during import
    genai = None  # type: ignore
    GoogleAPIError = Exception  # type: ignore

try:  # pragma: no cover - training metrics degrade gracefully without GPUs
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pynvml = None  # type: ignore

main
codex/integrate-pydantic-models-and-api-routes
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
 main
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:  # pragma: no cover - graceful degradation if dependency missing
    import google.generativeai as genai  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    genai = None  # type: ignore
    _GENAI_IMPORT_ERROR = "google-generativeai is not installed. Run 'pip install -r requirements.txt' to enable Gemini integration."
else:
    _GENAI_IMPORT_ERROR = None

# ---------------------------------------------------------------------------
# Storage directories
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
DATASETS_DIR = STORAGE_DIR / "datasets"
WORKFLOWS_DIR = STORAGE_DIR / "workflows"
LOGS_DIR = STORAGE_DIR / "logs"

for directory in (DATASETS_DIR, WORKFLOWS_DIR, LOGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

codex/rewrite-backend-using-fastapi-and-implement-routes-k68a2h

DATASETS_INDEX_PATH = DATASETS_DIR / "index.json"

main
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024
PREVIEW_LIMIT_BYTES = 1 * 1024 * 1024

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".csv"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | TEXT_EXTENSIONS
codex/rewrite-backend-using-fastapi-and-implement-routes-k68a2h

DATASET_METADATA_FILENAME = "metadata.json"
MAX_CONTEXT_SNIPPET_CHARS = 4000

main

TRAINING_STATE_PATH = STORAGE_DIR / "training_state.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger("workflow-backend")
LOGGER.setLevel(logging.INFO)

if not LOGGER.handlers:
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOGS_DIR / "backend.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(file_handler)

codex/rewrite-backend-using-fastapi-and-implement-routes-k68a2h
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if _GENAI_IMPORT_ERROR:
    LOGGER.warning(_GENAI_IMPORT_ERROR)


def _error_detail(message: str, *, error_code: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Helper to construct detail payloads for HTTP exceptions."""

    payload: Dict[str, Any] = {"message": message, "error_code": error_code}
    if details is not None:
        payload["details"] = details
    return payload


def _normalise_error(detail: Any) -> tuple[str, Optional[Dict[str, Any]]]:
    """Extract a message and detail payload from HTTPException.detail."""

    default_message = "An error occurred."
    message = default_message
    error_code: Optional[str] = None
    extra_details: Any = None

    if isinstance(detail, dict):
        message = str(detail.get("message") or default_message)
        error_code = detail.get("error_code") or detail.get("code")
        extra_details = detail.get("details")
    elif isinstance(detail, str):
        message = detail
    elif detail is not None:
        message = str(detail)

    combined_details: Optional[Dict[str, Any]]
    if error_code is None and extra_details is None:
        combined_details = None
    else:
        if isinstance(extra_details, dict):
            combined_details = dict(extra_details)
        elif extra_details is None:
            combined_details = {}
        else:
            combined_details = {"context": extra_details}

        if error_code:
            combined_details.setdefault("errorCode", error_code)

        if not combined_details:
            combined_details = None

    return message, combined_details


# ---------------------------------------------------------------------------
# Training job management
# ---------------------------------------------------------------------------


def _current_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


_GPU_LOCK = threading.Lock()
_GPU_INITIALISED = False


def _read_gpu_memory() -> Optional[int]:  # pragma: no cover - depends on GPU availability
    if pynvml is None:
        return None

    global _GPU_INITIALISED  # pylint: disable=global-statement
    try:
        with _GPU_LOCK:
            if not _GPU_INITIALISED:
                pynvml.nvmlInit()
                _GPU_INITIALISED = True
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    except Exception:
        return None
    return int(getattr(info, "used", 0))


class TrainingJob:
    """Represents a background training job that tracks progress and metrics."""

    def __init__(self, run_id: str, *, steps: int = 20, description: Optional[str] = None) -> None:
        self.run_id = run_id
        self.steps = max(1, steps)
        self.description = description or "Synthetic training job"
        self.log_path = LOGS_DIR / f"{run_id}.log"
        self._logger = logging.getLogger(f"{LOGGER.name}.train.{run_id}")
        self._logger.setLevel(logging.INFO)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name=f"training-{run_id}", daemon=True)
        self._handler: Optional[logging.Handler] = None

        now = _current_timestamp()
        self._snapshot: Dict[str, Any] = {
            "runId": self.run_id,
            "state": "pending",
            "progress": 0.0,
            "message": "Pending",
            "description": self.description,
            "startedAt": now,
            "updatedAt": now,
            "metrics": {
                "progressPercent": 0.0,
                "gpuMemoryBytes": _read_gpu_memory(),
                "logTail": "",
            },
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    def abort(self) -> None:
        self._stop_event.set()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._snapshot)

    def is_finished(self) -> bool:
        with self._lock:
            return self._snapshot["state"] in {"completed", "failed", "aborted"}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run(self) -> None:
        handler = logging.FileHandler(self.log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self._handler = handler
        self._logger.addHandler(handler)

        self._update_snapshot(state="running", message="Training started", progress=0.0)
        try:
            for step in range(self.steps):
                if self._stop_event.is_set():
                    self._update_snapshot(state="aborted", message="Training aborted", progress=self._progress_percent(step))
                    self._logger.info("Run %s aborted at step %s/%s", self.run_id, step, self.steps)
                    return

                time.sleep(0.5)
                step_number = step + 1
                progress = self._progress_percent(step_number)
                self._logger.info("Run %s completed step %s/%s", self.run_id, step_number, self.steps)
                self._update_snapshot(
                    message=f"Completed step {step_number} of {self.steps}",
                    progress=progress,
                )

            self._update_snapshot(state="completed", message="Training completed", progress=100.0)
        except Exception as exc:  # pragma: no cover - unexpected runtime errors
            self._logger.exception("Run %s failed: %s", self.run_id, exc)
            self._update_snapshot(state="failed", message=f"Training failed: {exc}")
        finally:
            if self._handler:
                self._logger.removeHandler(self._handler)
                self._handler.flush()
                self._handler.close()
                self._handler = None
            self._update_snapshot(force_log_refresh=True)

    def _progress_percent(self, step: int) -> float:
        if self.steps == 0:
            return 0.0
        return round(100.0 * min(step, self.steps) / self.steps, 2)

    def _update_snapshot(
        self,
        *,
        state: Optional[str] = None,
        message: Optional[str] = None,
        progress: Optional[float] = None,
        force_log_refresh: bool = False,
    ) -> None:
        with self._lock:
            if state is not None:
                self._snapshot["state"] = state
            if message is not None:
                self._snapshot["message"] = message
            if progress is not None:
                self._snapshot["progress"] = max(0.0, min(100.0, progress))
                self._snapshot["metrics"]["progressPercent"] = self._snapshot["progress"]

            self._snapshot["updatedAt"] = _current_timestamp()
            self._snapshot["metrics"]["gpuMemoryBytes"] = _read_gpu_memory()
            if force_log_refresh or self._snapshot["metrics"].get("logTail") == "":
                self._snapshot["metrics"]["logTail"] = self._read_log_tail()

    def _read_log_tail(self, max_lines: int = 20) -> str:
        try:
            with self.log_path.open("r", encoding="utf-8", errors="ignore") as log_file:
                lines = log_file.readlines()[-max_lines:]
        except FileNotFoundError:
            return ""
        return "".join(lines).strip()


class TrainingJobManager:
    """Simple in-memory registry for background training jobs."""

    def __init__(self) -> None:
        self._jobs: Dict[str, TrainingJob] = {}
        self._lock = threading.Lock()

    def start_job(self, *, steps: int = 20, description: Optional[str] = None) -> str:
        run_id = str(uuid4())
        job = TrainingJob(run_id, steps=steps, description=description)
        with self._lock:
            self._jobs[run_id] = job
        job.start()
        return run_id

    def get_snapshot(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(run_id)
        if job is None:
            return None
        snapshot = job.snapshot()
        if job.is_finished():
            snapshot = self._ensure_log_tail(snapshot)
        return snapshot

    def abort_job(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(run_id)
        if job is None:
            return None
        job.abort()
        return job.snapshot()

    def _ensure_log_tail(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        metrics = snapshot.setdefault("metrics", {})
        if metrics.get("logTail"):
            return snapshot
        run_id = snapshot.get("runId")
        if not run_id:
            return snapshot
        log_path = LOGS_DIR / f"{run_id}.log"
        try:
            with log_path.open("r", encoding="utf-8", errors="ignore") as log_file:
                lines = log_file.readlines()[-20:]
                metrics["logTail"] = "".join(lines).strip()
        except FileNotFoundError:
            metrics.setdefault("logTail", "")
        return snapshot


TRAINING_MANAGER = TrainingJobManager()


# ---------------------------------------------------------------------------
# FastAPI setup
# ---------------------------------------------------------------------------
class GeminiService:
    """Thin wrapper around the google-generativeai client."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger
        self._models: Dict[str, Any] = {}
        self.available = False
        self.error_message: Optional[str] = None

        if genai is None:
            self.error_message = _GENAI_IMPORT_ERROR
            return

        if not GEMINI_API_KEY:
            self.error_message = "GEMINI_API_KEY environment variable is not set; Gemini integration is disabled."
            self._logger.warning(self.error_message)
            return

        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.available = True
            self._logger.info("Gemini client initialised successfully.")
        except Exception as exc:  # pragma: no cover - configuration failure is environment specific
            self.error_message = f"Failed to configure Gemini client: {exc}"
            self._logger.error(self.error_message)

    def generate(self, model_name: str, prompt: str, context: Dict[str, Any]) -> str:
        if not self.available or genai is None:
            raise RuntimeError(self.error_message or "Gemini service is not available.")

        try:
            model = self._models.get(model_name)
            if model is None:
                model = genai.GenerativeModel(model_name)
                self._models[model_name] = model

            parts = [prompt]
            if context:
                context_json = json.dumps(context, ensure_ascii=False, indent=2)
                if len(context_json) > MAX_CONTEXT_SNIPPET_CHARS:
                    context_json = context_json[: MAX_CONTEXT_SNIPPET_CHARS - 1] + "…"
                parts.append("Workflow inputs:\n" + context_json)

            response = model.generate_content(parts)
            text = getattr(response, "text", None)
            if text:
                return text

            # Fallback: scan candidates for the first textual part
            candidates = getattr(response, "candidates", [])
            for candidate in candidates or []:
                content = getattr(candidate, "content", None)
                if not content:
                    continue
                for part in getattr(content, "parts", []) or []:
                    text_part = getattr(part, "text", None)
                    if text_part:
                        return text_part

            raise RuntimeError("Gemini response did not include any text output.")
        except Exception as exc:  # pragma: no cover - dependency raises runtime errors
            raise RuntimeError(f"Gemini generation failed: {exc}") from exc


GEMINI_SERVICE = GeminiService(LOGGER)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_DEFAULT_MODEL = os.getenv("GEMINI_DEFAULT_MODEL", "gemini-1.5-flash")
_GEMINI_CONFIGURED = False

if genai is None:
    LOGGER.warning("google-generativeai package is unavailable; Gemini functionality is disabled.")
else:
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            _GEMINI_CONFIGURED = True
            LOGGER.info("Gemini client configured successfully.")
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.error("Failed to configure Gemini client: %s", exc)
    else:
        LOGGER.warning("GEMINI_API_KEY is not set; Gemini functionality is disabled.")
main

# ---------------------------------------------------------------------------
# FastAPI setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Workflow Orchestrator API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:  # pragma: no cover - exercised via API
    message, details = _normalise_error(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.status_code, "message": message, "details": details}},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover - defensive guard
    LOGGER.exception("Unhandled error while processing request: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error.",
                "details": {"errorCode": "INTERNAL_SERVER_ERROR"},
            }
        },
    )

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class Position(BaseModel):
    x: float
    y: float


class WorkflowNode(BaseModel):
    id: str
    type: str
    position: Position
    data: Dict[str, Any] = Field(default_factory=dict)


class WorkflowEdge(BaseModel):
    id: str
    fromNode: str
    fromPort: str
    toNode: str
    toPort: str


class WorkflowDefinition(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    nodes: List[WorkflowNode]
    edges: List[WorkflowEdge]


class WorkflowRunNode(BaseModel):
    id: str
    type: str
    x: float
    y: float
    data: Dict[str, Any]
    status: str


class WorkflowRunResponse(BaseModel):
    run_id: str
    nodes: List[WorkflowRunNode]


codex/integrate-pydantic-models-and-api-routes
class TrainingStartRequest(BaseModel):
    dataset_id: str = Field(..., alias="datasetId")
    model_id: str = Field(..., alias="modelId")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None

    class Config:
        allow_population_by_field_name = True


class TrainingStatusResponse(BaseModel):
    job_id: str = Field(..., alias="jobId")
    dataset_id: str = Field(..., alias="datasetId")
    model_id: str = Field(..., alias="modelId")
    status: str
    progress: float = Field(..., ge=0.0, le=1.0)
    gpu_memory_mb: Optional[int] = Field(None, alias="gpuMemoryMB")
    log_tail: List[str] = Field(default_factory=list, alias="logTail")
    submitted_at: str = Field(..., alias="submittedAt")
    started_at: Optional[str] = Field(None, alias="startedAt")
    completed_at: Optional[str] = Field(None, alias="completedAt")
    parameters: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        allow_population_by_field_name = True


class TrainingAbortRequest(BaseModel):
    job_id: Optional[str] = Field(None, alias="jobId")

    class Config:
        allow_population_by_field_name = True

class TrainStartRequest(BaseModel):
    steps: int = Field(20, ge=1, le=10_000)
    description: Optional[str] = None


class TrainStartResponse(BaseModel):
    runId: str


class TrainStatusResponse(BaseModel):
    runId: str
    state: str
    progress: float
    message: str
    description: Optional[str] = None
    startedAt: str
    updatedAt: str
    metrics: Dict[str, Any]


class TrainAbortRequest(BaseModel):
    runId: str
main


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("Failed to read JSON from %s. Returning default.", path)
        return default


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
codex/rewrite-backend-using-fastapi-and-implement-routes-k68a2h



def _dataset_index() -> List[Dict[str, Any]]:
    return _load_json(DATASETS_INDEX_PATH, [])


def _write_dataset_index(entries: List[Dict[str, Any]]) -> None:
    _save_json(DATASETS_INDEX_PATH, entries)
main


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


codex/rewrite-backend-using-fastapi-and-implement-routes-k68a2h
def _dataset_dir(dataset_id: str) -> Path:
    return DATASETS_DIR / dataset_id


def _dataset_metadata_path(dataset_id: str) -> Path:
    return _dataset_dir(dataset_id) / DATASET_METADATA_FILENAME


def _dataset_source_file(dataset_folder: Path) -> Optional[Path]:
    for candidate in dataset_folder.iterdir():
        if candidate.is_file() and candidate.name != DATASET_METADATA_FILENAME:
            return candidate
    return None


def _normalise_filename(filename: str) -> str:
    return Path(filename).name


def _collect_dataset_metadata() -> List[Dict[str, Any]]:
    datasets: List[Dict[str, Any]] = []
    for dataset_folder in DATASETS_DIR.iterdir():
        if not dataset_folder.is_dir():
            continue

        dataset_id = dataset_folder.name
        metadata_path = _dataset_metadata_path(dataset_id)
        metadata = _load_json(metadata_path, None)

        source_file = _dataset_source_file(dataset_folder)
        if not metadata:
            if not source_file:
                continue
            metadata = {
                "id": dataset_id,
                "datasetId": dataset_id,
                "name": source_file.name,
                "storedFilename": source_file.name,
                "size": source_file.stat().st_size,
                "type": _detect_dataset_type(source_file.name),
                "mimeType": _detect_mime_type(source_file.suffix),
                "preview": None,
                "uploadedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(source_file.stat().st_mtime)),
                "storagePath": str(source_file.relative_to(BASE_DIR)),
            }
            _save_json(metadata_path, metadata)
        else:
            metadata.setdefault("id", dataset_id)
            metadata.setdefault("datasetId", dataset_id)
            metadata.setdefault("name", metadata.get("storedFilename", metadata.get("name", dataset_id)))
            metadata.setdefault("storedFilename", metadata.get("name", dataset_id))
            metadata.setdefault("type", _detect_dataset_type(metadata.get("name", dataset_id)))
            metadata.setdefault("mimeType", _detect_mime_type(Path(metadata.get("storedFilename", "")).suffix))
            if source_file and not metadata.get("storagePath"):
                metadata["storagePath"] = str(source_file.relative_to(BASE_DIR))
            elif metadata.get("storagePath"):
                storage_path = metadata["storagePath"]
                resolved_path = BASE_DIR / storage_path
                if not resolved_path.exists() and source_file:
                    metadata["storagePath"] = str(source_file.relative_to(BASE_DIR))
            metadata.setdefault(
                "uploadedAt",
                time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(metadata_path.stat().st_mtime if metadata_path.exists() else time.time()),
                ),
            )

        datasets.append(metadata)

    datasets.sort(key=lambda item: item.get("uploadedAt", ""), reverse=True)
    return datasets

def _format_inputs_for_prompt(inputs: Dict[str, Any]) -> str:
    if not inputs:
        return "No inputs were provided."
    try:
        return json.dumps(inputs, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        serialisable = {key: repr(value) for key, value in inputs.items()}
        return json.dumps(serialisable, ensure_ascii=False, indent=2)


def _generate_gemini_report(model_id: str, prompt: str) -> str:
    if genai is None or not _GEMINI_CONFIGURED:
        raise RuntimeError("Gemini client is not configured. Please set the GEMINI_API_KEY environment variable.")

    try:
        model = genai.GenerativeModel(model_id)
        response = model.generate_content(prompt)
    except GoogleAPIError as exc:  # type: ignore[arg-type]
        raise RuntimeError(f"Gemini API error: {exc}") from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"Gemini request failed: {exc}") from exc

    text_response = getattr(response, "text", None)
    if text_response:
        return text_response.strip()

    candidate_texts: List[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            continue
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                candidate_texts.append(part_text)

    if candidate_texts:
        return "\n".join(candidate_texts).strip()

    raise RuntimeError("Gemini response did not contain any text output.")
main


# ---------------------------------------------------------------------------
# Training controller
# ---------------------------------------------------------------------------


class TrainingController:
    """Simple controller that tracks the lifecycle of a single training job."""

    def __init__(self, state_path: Path, logger: logging.Logger) -> None:
        self._state_path = state_path
        self._logger = logger
        self._lock = threading.Lock()
        self._state = self._load_state()

    @staticmethod
    def _now_iso() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _default_state(self) -> Dict[str, Any]:
        return {"jobs": {}, "active_job_id": None}

    def _load_state(self) -> Dict[str, Any]:
        state = _load_json(self._state_path, self._default_state())
        if not isinstance(state, dict):
            state = self._default_state()
        state.setdefault("jobs", {})
        state.setdefault("active_job_id", None)
        return state

    def _refresh_state(self) -> None:
        self._state = self._load_state()

    def _persist_state(self) -> None:
        _save_json(self._state_path, self._state)

    @staticmethod
    def _append_log(job: Dict[str, Any], message: str) -> None:
        logs = job.setdefault("logTail", [])
        if message not in logs:
            logs.append(message)

    def _simulate_progress(self, job: Dict[str, Any]) -> None:
        if job.get("status") != "running":
            return

        started_epoch = job.get("_startedAtEpoch")
        if started_epoch is None:
            started_iso = job.get("startedAt") or job.get("submittedAt")
            try:
                started_epoch = time.mktime(time.strptime(str(started_iso), "%Y-%m-%dT%H:%M:%SZ"))
            except (TypeError, ValueError):
                started_epoch = time.time()
            job["_startedAtEpoch"] = started_epoch

        elapsed = max(0.0, time.time() - float(started_epoch))
        simulated_duration = 60.0
        progress = min(1.0, elapsed / simulated_duration)
        job["progress"] = progress
        job["gpuMemoryMB"] = job.get("gpuMemoryMB") or 6144
        self._append_log(job, "Training job is running.")

        if progress >= 1.0:
            job["status"] = "completed"
            job["completedAt"] = job.get("completedAt") or self._now_iso()
            self._append_log(job, "Training completed successfully.")
            if self._state.get("active_job_id") == job.get("jobId"):
                self._state["active_job_id"] = None

    def _sanitise_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        payload = {key: value for key, value in job.items() if not key.startswith("_")}
        payload.setdefault("progress", 0.0)
        payload.setdefault("gpuMemoryMB", None)
        payload.setdefault("logTail", [])
        payload.setdefault("parameters", {})
        return payload

    def _build_response(self, job: Dict[str, Any]) -> TrainingStatusResponse:
        payload = self._sanitise_job(job)
        return TrainingStatusResponse.parse_obj(payload)

    def start(self, request: TrainingStartRequest) -> TrainingStatusResponse:
        with self._lock:
            self._refresh_state()
            active_job_id = self._state.get("active_job_id")
            if active_job_id:
                active_job = self._state.get("jobs", {}).get(active_job_id)
                if active_job and active_job.get("status") in {"queued", "running"}:
                    raise HTTPException(
                        status_code=409,
                        detail=_error_detail(
                            "A training job is already running.",
                            error_code="TRAINING_JOB_ACTIVE",
                            details={"jobId": active_job_id},
                        ),
                    )

            job_id = str(uuid4())
            submitted_at = self._now_iso()
            job: Dict[str, Any] = {
                "jobId": job_id,
                "datasetId": request.dataset_id,
                "modelId": request.model_id,
                "parameters": dict(request.hyperparameters),
                "notes": request.notes,
                "status": "running",
                "progress": 0.0,
                "gpuMemoryMB": None,
                "logTail": ["Training job submitted.", "Initialising training run..."],
                "submittedAt": submitted_at,
                "startedAt": submitted_at,
                "completedAt": None,
                "_startedAtEpoch": time.time(),
            }

            self._state.setdefault("jobs", {})[job_id] = job
            self._state["active_job_id"] = job_id
            self._persist_state()
            self._logger.info(
                "Training job %s started with dataset %s and model %s", job_id, request.dataset_id, request.model_id
            )
            return self._build_response(job)

    def status(self, job_id: Optional[str]) -> TrainingStatusResponse:
        with self._lock:
            self._refresh_state()
            target_job_id = job_id or self._state.get("active_job_id")
            if not target_job_id:
                raise HTTPException(
                    status_code=404,
                    detail=_error_detail(
                        "No training job is currently active.",
                        error_code="TRAINING_JOB_NOT_FOUND",
                    ),
                )

            job = self._state.get("jobs", {}).get(target_job_id)
            if not job:
                raise HTTPException(
                    status_code=404,
                    detail=_error_detail(
                        "Training job not found.",
                        error_code="TRAINING_JOB_NOT_FOUND",
                        details={"jobId": target_job_id},
                    ),
                )

            self._simulate_progress(job)
            self._state["jobs"][target_job_id] = job
            self._persist_state()
            return self._build_response(job)

    def abort(self, job_id: Optional[str]) -> TrainingStatusResponse:
        with self._lock:
            self._refresh_state()
            target_job_id = job_id or self._state.get("active_job_id")
            if not target_job_id:
                raise HTTPException(
                    status_code=404,
                    detail=_error_detail(
                        "No training job is currently active.",
                        error_code="TRAINING_JOB_NOT_FOUND",
                    ),
                )

            job = self._state.get("jobs", {}).get(target_job_id)
            if not job:
                raise HTTPException(
                    status_code=404,
                    detail=_error_detail(
                        "Training job not found.",
                        error_code="TRAINING_JOB_NOT_FOUND",
                        details={"jobId": target_job_id},
                    ),
                )

            self._simulate_progress(job)

            if job.get("status") in {"completed", "aborted"}:
                return self._build_response(job)

            job["status"] = "aborted"
            job["completedAt"] = self._now_iso()
            job["progress"] = min(1.0, float(job.get("progress", 0.0)))
            self._append_log(job, "Training aborted by user request.")

            if self._state.get("active_job_id") == target_job_id:
                self._state["active_job_id"] = None

            self._state["jobs"][target_job_id] = job
            self._persist_state()
            self._logger.info("Training job %s aborted by request.", target_job_id)
            return self._build_response(job)


TRAINING_CONTROLLER = TrainingController(TRAINING_STATE_PATH, LOGGER)


# ---------------------------------------------------------------------------
# Dataset management
# ---------------------------------------------------------------------------


@app.get("/api/v1/models")
def list_models() -> List[Dict[str, str]]:
    """Return a static list of available models."""

    return [
        {
            "id": "gemini-1.5-flash",
            "name": "Gemini 1.5 Flash",
            "description": "Fast multimodal model for interactive workflows.",
        },
        {
            "id": "gemini-1.5-pro",
            "name": "Gemini 1.5 Pro",
            "description": "Higher quality Gemini model suited for complex reasoning tasks.",
        },
    ]


@app.get("/api/v1/datasets")
def list_datasets() -> List[Dict[str, Any]]:
    """Return metadata for uploaded datasets."""

codex/rewrite-backend-using-fastapi-and-implement-routes-k68a2h
    return _collect_dataset_metadata()

    return _dataset_index()
main


@app.post("/api/v1/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail=_error_detail(
                "Uploaded file must include a filename.",
                error_code="DATASET_MISSING_FILENAME",
            ),
        )

    original_filename = _normalise_filename(file.filename)
    extension = Path(original_filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=_error_detail(
                "File type is not supported.",
                error_code="DATASET_UNSUPPORTED_TYPE",
                details={"allowedExtensions": sorted(ALLOWED_EXTENSIONS)},
            ),
        )

    content = await file.read()
    size_bytes = len(content)

    if size_bytes == 0:
        raise HTTPException(
            status_code=400,
            detail=_error_detail(
                "Uploaded file is empty.",
                error_code="DATASET_EMPTY_FILE",
            ),
        )
    if size_bytes > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=_error_detail(
                "File exceeds maximum allowed size of 20MB.",
                error_code="DATASET_FILE_TOO_LARGE",
                details={"maxBytes": MAX_UPLOAD_SIZE_BYTES},
            ),
        )

    dataset_id = str(uuid4())
codex/rewrite-backend-using-fastapi-and-implement-routes-k68a2h
    dataset_folder = _dataset_dir(dataset_id)
    dataset_folder.mkdir(parents=True, exist_ok=False)

    stored_filename = original_filename or f"dataset{extension}"
    stored_path = dataset_folder / stored_filename
    stored_path.write_bytes(content)

    preview: Optional[str] = None
    if extension in IMAGE_EXTENSIONS and size_bytes <= PREVIEW_LIMIT_BYTES:
        encoded = base64.b64encode(content).decode("utf-8")
        preview = f"data:{_detect_mime_type(extension)};base64,{encoded}"
    elif extension in TEXT_EXTENSIONS:
        try:
            preview = content[:PREVIEW_LIMIT_BYTES].decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - defensive decode guard
            preview = None

    uploaded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    storage_path = str(stored_path.relative_to(BASE_DIR))

    metadata = {
        "id": dataset_id,
        "datasetId": dataset_id,
        "name": original_filename or stored_filename,
        "storedFilename": stored_filename,
        "size": size_bytes,
        "type": _detect_dataset_type(original_filename),
        "mimeType": _detect_mime_type(extension),
        "preview": preview,
        "uploadedAt": uploaded_at,
        "storagePath": storage_path,
    }

    _save_json(_dataset_metadata_path(dataset_id), metadata)

    LOGGER.info("Uploaded dataset %s (%s) -> %s", original_filename, dataset_id, storage_path)

    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=_error_detail(
                "Unsupported file type.",
                error_code="DATASET_UNSUPPORTED_TYPE",
                details={
                    "allowedExtensions": sorted(ALLOWED_EXTENSIONS),
                    "message": f"Allowed extensions: {allowed}",
                },
            ),
        )
    stored_name = f"{dataset_id}{extension}"
    stored_path = DATASETS_DIR / stored_name
    stored_path.write_bytes(content)

    preview: Optional[str] = None
    if extension.lower() in IMAGE_EXTENSIONS and size_bytes <= PREVIEW_LIMIT_BYTES:
        encoded = base64.b64encode(content).decode("utf-8")
        preview = f"data:{_detect_mime_type(extension)};base64,{encoded}"

    metadata = {
        "id": dataset_id,
        "name": file.filename,
        "filename": stored_name,
        "path": str(Path("datasets") / stored_name),
        "size": size_bytes,
        "type": _detect_dataset_type(file.filename),
        "mimeType": _detect_mime_type(extension),
        "preview": preview,
        "uploadedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    entries = _dataset_index()
    entries.append(metadata)
    _write_dataset_index(entries)

    LOGGER.info("Uploaded dataset %s (%s)", file.filename, dataset_id)
main
    return metadata


# ---------------------------------------------------------------------------
# Workflow persistence
# ---------------------------------------------------------------------------


@app.get("/api/v1/workflows")
def list_workflows() -> List[Dict[str, Any]]:
    """Return metadata for saved workflows."""

    workflows: List[Dict[str, Any]] = []
    for workflow_file in WORKFLOWS_DIR.glob("*.json"):
        data = _load_json(workflow_file, {})
        workflows.append(
            {
                "id": workflow_file.stem,
                "name": data.get("name") or f"Workflow {workflow_file.stem[:8]}",
                "description": data.get("description", ""),
                "updatedAt": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(workflow_file.stat().st_mtime),
                ),
            }
        )
    return workflows


@app.get("/api/v1/workflows/{workflow_id}")
def get_workflow(workflow_id: str) -> Dict[str, Any]:
    workflow_path = WORKFLOWS_DIR / f"{workflow_id}.json"
    if not workflow_path.exists():
        raise HTTPException(
            status_code=404,
            detail=_error_detail(
                "Workflow not found.",
                error_code="WORKFLOW_NOT_FOUND",
                details={"workflowId": workflow_id},
            ),
        )
    return _load_json(workflow_path, {})


@app.post("/api/v1/workflows/save")
def save_workflow(definition: WorkflowDefinition) -> Dict[str, str]:
    workflow_id = definition.id or str(uuid4())
    payload = definition.dict()
    payload["id"] = workflow_id

    workflow_path = WORKFLOWS_DIR / f"{workflow_id}.json"
    _save_json(workflow_path, payload)

    LOGGER.info("Saved workflow %s", workflow_id)
    return {"id": workflow_id}


# ---------------------------------------------------------------------------
# Training endpoints
# ---------------------------------------------------------------------------


@app.post("/api/v1/train/start", response_model=TrainingStatusResponse)
def start_training_job(request: TrainingStartRequest) -> TrainingStatusResponse:
    """Start a new training job for a dataset/model pair."""

    return TRAINING_CONTROLLER.start(request)


@app.get("/api/v1/train/status", response_model=TrainingStatusResponse)
def get_training_status(job_id: Optional[str] = Query(default=None, alias="jobId")) -> TrainingStatusResponse:
    """Return the status of the requested or active training job."""

    return TRAINING_CONTROLLER.status(job_id)


@app.post("/api/v1/train/abort", response_model=TrainingStatusResponse)
def abort_training_job(request: TrainingAbortRequest) -> TrainingStatusResponse:
    """Abort the active training job or the job identified in the payload."""

    return TRAINING_CONTROLLER.abort(request.job_id)


# ---------------------------------------------------------------------------
# Workflow execution
# ---------------------------------------------------------------------------


class WorkflowExecutor:
    """Minimal workflow executor that processes nodes in topological order."""

    def __init__(
        self,
        definition: WorkflowDefinition,
        *,
        gemini_service: Optional[GeminiService] = None,
        logger: Optional[logging.Logger] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self.definition = definition
        self.nodes = {node.id: node for node in definition.nodes}
        self.node_outputs: Dict[str, Dict[str, Any]] = {}
        self.node_states: Dict[str, WorkflowRunNode] = {
            node.id: WorkflowRunNode(
                id=node.id,
                type=node.type,
                x=node.position.x,
                y=node.position.y,
                data=dict(node.data),
                status="pending",
            )
            for node in definition.nodes
        }
        self.adjacency: Dict[str, List[str]] = defaultdict(list)
        self.incoming_edges: Dict[str, List[WorkflowEdge]] = defaultdict(list)
        self._build_graph(definition.edges)
        self.gemini_service = gemini_service
        self.logger = logger or LOGGER
        self.run_id = run_id or str(uuid4())

    def _build_graph(self, edges: List[WorkflowEdge]) -> None:
        indegree = defaultdict(int)
        for node_id in self.nodes:
            indegree[node_id] = 0

        for edge in edges:
            if edge.fromNode not in self.nodes or edge.toNode not in self.nodes:
                raise HTTPException(
                    status_code=400,
                    detail=_error_detail(
                        f"Edge {edge.id} references unknown nodes.",
                        error_code="WORKFLOW_INVALID_EDGE",
                        details={
                            "edgeId": edge.id,
                            "fromNode": edge.fromNode,
                            "toNode": edge.toNode,
                        },
                    ),
                )
            self.adjacency[edge.fromNode].append(edge.toNode)
            self.incoming_edges[edge.toNode].append(edge)
            indegree[edge.toNode] += 1

        queue: deque[str] = deque([node_id for node_id, degree in indegree.items() if degree == 0])
        if not queue and self.nodes:
            raise HTTPException(
                status_code=400,
                detail=_error_detail(
                    "Workflow has no entry nodes (cycle detected).",
                    error_code="WORKFLOW_NO_ENTRY_NODES",
                ),
            )

        execution_order: List[str] = []
        while queue:
            current = queue.popleft()
            execution_order.append(current)
            for neighbor in self.adjacency[current]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        if len(execution_order) != len(self.nodes):
            raise HTTPException(
                status_code=400,
                detail=_error_detail(
                    "Workflow contains cycles and cannot be executed.",
                    error_code="WORKFLOW_CONTAINS_CYCLE",
                ),
            )

        self.execution_order = execution_order

    def run(self) -> WorkflowRunResponse:
        if not hasattr(self, "logger"):
            self.logger = LOGGER
        if not hasattr(self, "run_id"):
            self.run_id = str(uuid4())

        self.logger.info(
            "Run %s starting execution of %d nodes", self.run_id, len(self.execution_order)
        )
        for node_id in self.execution_order:
            node = self.nodes[node_id]
            state = self.node_states[node_id]
            state.status = "running"
            self.logger.info(
                "Run %s starting node %s (%s)", self.run_id, node.id, node.type
            )
            inputs = self._collect_inputs(node_id)
            try:
                outputs, updated_data = self._execute_node(node, inputs)
                state.data = updated_data
                state.status = "done"
                self.node_outputs[node_id] = outputs
                self.logger.info(
                    "Run %s completed node %s (%s)", self.run_id, node.id, node.type
                )
            except Exception as exc:  # pylint: disable=broad-except
                state.status = "failed"
                state.data = {**state.data, "error": str(exc)}
                self.node_outputs[node_id] = {}
                self.logger.exception(
                    "Run %s node %s failed: %s", self.run_id, node.id, exc
                )

        ordered_nodes = [self.node_states[node.id] for node in self.definition.nodes]
        self.logger.info("Run %s completed", self.run_id)
        return WorkflowRunResponse(run_id=self.run_id, nodes=ordered_nodes)

    def _collect_inputs(self, node_id: str) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {}
        for edge in self.incoming_edges.get(node_id, []):
            upstream_outputs = self.node_outputs.get(edge.fromNode, {})
            if edge.fromPort in upstream_outputs:
                inputs[edge.toPort] = upstream_outputs[edge.fromPort]
        return inputs

    @staticmethod
    def _evaluate_decision_condition(condition: str, input_value: Any) -> bool:
        """Evaluate a decision condition using a constrained AST interpreter."""

        try:
            tree = ast.parse(condition, mode="eval")
        except SyntaxError as exc:  # pragma: no cover - syntax errors handled as ValueError
            raise ValueError("invalid syntax") from exc

        def eval_operand(node: ast.AST) -> Any:
            if isinstance(node, ast.Name) and node.id == "input":
                return input_value
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float, str, bool, type(None))):
                    return node.value
                raise ValueError("unsupported constant type")
            raise ValueError("unsupported expression element")

        def eval_node(node: ast.AST) -> bool:
            if isinstance(node, ast.Expression):
                return eval_node(node.body)
            if isinstance(node, ast.BoolOp):
                values = [eval_node(value) for value in node.values]
                if isinstance(node.op, ast.And):
                    return all(values)
                if isinstance(node.op, ast.Or):
                    return any(values)
                raise ValueError("unsupported boolean operator")
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                return not eval_node(node.operand)
            if isinstance(node, ast.Compare):
                left = eval_operand(node.left)
                result = True
                for operator, comparator in zip(node.ops, node.comparators):
                    right = eval_operand(comparator)
                    if isinstance(operator, ast.Eq):
                        comparison = left == right
                    elif isinstance(operator, ast.NotEq):
                        comparison = left != right
                    elif isinstance(operator, ast.Lt):
                        comparison = left < right
                    elif isinstance(operator, ast.LtE):
                        comparison = left <= right
                    elif isinstance(operator, ast.Gt):
                        comparison = left > right
                    elif isinstance(operator, ast.GtE):
                        comparison = left >= right
                    else:
                        raise ValueError("unsupported comparison operator")
                    result = result and bool(comparison)
                    left = right
                return result
            if isinstance(node, ast.Name) and node.id == "input":
                return bool(input_value)
            raise ValueError("unsupported expression element")

        result = eval_node(tree)
        if not isinstance(result, bool):
            result = bool(result)
        return result

    def _execute_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Placeholder node execution logic."""

        data = dict(node.data)
        if node.type == "generate_report":
codex/rewrite-backend-using-fastapi-and-implement-routes-k68a2h
            prompt = data.get("prompt") or "Summarise the provided workflow inputs into a concise report."
            model_name = data.get("modelId") or data.get("model") or "gemini-1.5-flash"

            if self.gemini_service and self.gemini_service.available:
                report = self.gemini_service.generate(model_name, prompt, inputs)
            else:
                error_message = (
                    self.gemini_service.error_message
                    if self.gemini_service and self.gemini_service.error_message
                    else "Gemini service is not configured."
                )
                raise RuntimeError(error_message)

            data["report"] = report

            prompt = data.get(
                "prompt",
                "Generate a detailed report that summarises the provided workflow inputs.",
            )
            context = data.get("context")
            formatted_inputs = _format_inputs_for_prompt(inputs)
            prompt_sections = [prompt]
            if context:
                prompt_sections.append(str(context))
            prompt_sections.extend(["Workflow inputs:", formatted_inputs])
            final_prompt = "\n\n".join(prompt_sections)

            model_id = data.get("model") or data.get("modelId") or GEMINI_DEFAULT_MODEL
            try:
                report = _generate_gemini_report(model_id, final_prompt)
            except Exception as exc:  # pylint: disable=broad-except
                raise RuntimeError(f"Gemini report generation failed: {exc}") from exc

            data["report"] = report
            data["model"] = model_id
            data["aggregated_inputs"] = formatted_inputs
main
            return {"out": report}, data

        if node.type == "decision_logic":
            condition = data.get("condition", "bool(input)")
            input_value = inputs.get("in")
            try:
                decision = self._evaluate_decision_condition(condition, input_value)
            except ValueError as exc:
                self.logger.warning(
                    "Run %s decision node %s failed to evaluate condition '%s': %s",
                    self.run_id,
                    node.id,
                    condition,
                    exc,
                )
                decision = bool(input_value)
            data["decision"] = decision
            return {"true": input_value if decision else None, "false": None if decision else input_value}, data

        if node.type in {"image_input", "text_input", "data_hub", "model_hub"}:
            return {"out": data}, data

        if node.type == "image_classifier":
            result = {
                "label": "unknown",
                "confidence": 0.0,
                "received": bool(inputs.get("image")),
                "model": inputs.get("model"),
            }
            data["classification"] = result
            return {"out": result}, data

        # Default passthrough behaviour for any other node types
        return inputs or {"out": data}, data


@app.post("/api/v1/workflow/run", response_model=WorkflowRunResponse)
def run_workflow(definition: WorkflowDefinition) -> WorkflowRunResponse:
    if not definition.nodes:
        raise HTTPException(
            status_code=400,
            detail=_error_detail(
                "Workflow must contain at least one node.",
                error_code="WORKFLOW_EMPTY",
            ),
        )

    run_id = str(uuid4())
    log_path = LOGS_DIR / f"{run_id}.log"
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    run_logger = logging.getLogger(f"{LOGGER.name}.run.{run_id}")
    run_logger.setLevel(logging.INFO)
    run_logger.addHandler(handler)

    executor = WorkflowExecutor(
        definition,
        gemini_service=GEMINI_SERVICE,
        logger=run_logger,
        run_id=run_id,
    )

    try:
        run_logger.info("Run %s started", run_id)
        response = executor.run()
        return response
    finally:
        run_logger.info("Run %s finished", run_id)
        run_logger.removeHandler(handler)
        handler.close()


# ---------------------------------------------------------------------------
# Training endpoints
# ---------------------------------------------------------------------------


@app.post("/api/v1/train/start", response_model=TrainStartResponse)
def start_training_job(payload: TrainStartRequest = Body(default=TrainStartRequest())) -> TrainStartResponse:
    run_id = TRAINING_MANAGER.start_job(steps=payload.steps, description=payload.description)
    LOGGER.info("Started training run %s", run_id)
    return TrainStartResponse(runId=run_id)


@app.get("/api/v1/train/status", response_model=TrainStatusResponse)
def get_training_status(run_id: str) -> TrainStatusResponse:
    snapshot = TRAINING_MANAGER.get_snapshot(run_id)
    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail=_error_detail(
                "Training run not found.",
                error_code="TRAINING_RUN_NOT_FOUND",
                details={"runId": run_id},
            ),
        )
    return TrainStatusResponse(**snapshot)


@app.post("/api/v1/train/abort", response_model=TrainStatusResponse)
def abort_training_job(payload: TrainAbortRequest) -> TrainStatusResponse:
    snapshot = TRAINING_MANAGER.abort_job(payload.runId)
    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail=_error_detail(
                "Training run not found.",
                error_code="TRAINING_RUN_NOT_FOUND",
                details={"runId": payload.runId},
            ),
        )
    return TrainStatusResponse(**snapshot)
