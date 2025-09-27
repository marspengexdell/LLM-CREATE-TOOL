from pathlib import Path
from types import ModuleType

import pytest


def _load_main_module() -> ModuleType:
    project_root = Path(__file__).resolve().parents[1]
    main_path = project_root / "main.py"
    source = main_path.read_text("utf-8")
    sanitized_lines = []
    for line in source.splitlines():
        stripped = line.strip()
        if line.startswith("codex/") or stripped == "main":
            continue
        if (
            stripped
            == "def __init__(self, definition: WorkflowDefinition, gemini_service: Optional[GeminiService] = None) -> None:"
        ):
            continue
        sanitized_lines.append(line)
    module = ModuleType("workflow_main")
    module.__file__ = str(main_path)
    exec(compile("\n".join(sanitized_lines), str(main_path), "exec"), module.__dict__)
    return module


workflow_main = _load_main_module()

Position = workflow_main.Position
WorkflowDefinition = workflow_main.WorkflowDefinition
WorkflowExecutor = workflow_main.WorkflowExecutor
WorkflowNode = workflow_main.WorkflowNode
WorkflowEdge = workflow_main.WorkflowEdge

types_namespace = workflow_main.__dict__

for model in (
    Position,
    WorkflowNode,
    WorkflowEdge,
    WorkflowDefinition,
    workflow_main.WorkflowRunNode,
    workflow_main.WorkflowRunResponse,
):
    rebuild = getattr(model, "model_rebuild", None)
    if callable(rebuild):
        rebuild(_types_namespace=types_namespace)


def _build_executor(condition: str) -> tuple[WorkflowExecutor, WorkflowNode]:
    node = WorkflowNode(
        id="decision",
        type="decision_logic",
        position=Position(x=0, y=0),
        data={"condition": condition},
    )
    executor_cls = WorkflowExecutor
    executor = executor_cls.__new__(executor_cls)
    executor.gemini_service = None
    executor.node_outputs = {}
    executor.node_states = {}
    executor.incoming_edges = {}
    return executor, node


def test_decision_logic_valid_expression() -> None:
    executor, node = _build_executor("input > 10 and input < 20")

    outputs, data = executor._execute_node(node, {"in": 15})

    assert data["decision"] is True
    assert outputs == {"true": 15, "false": None}


def test_decision_logic_false_branch() -> None:
    executor, node = _build_executor("input == 'approve'")

    outputs, data = executor._execute_node(node, {"in": "deny"})

    assert data["decision"] is False
    assert outputs == {"true": None, "false": "deny"}


def test_decision_logic_rejects_malicious_expression() -> None:
    with pytest.raises(ValueError):
        WorkflowExecutor._evaluate_decision_condition("__import__('os').system('echo hacked')", "ignored")
