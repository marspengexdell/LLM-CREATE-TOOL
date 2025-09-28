import json
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("torch", reason="training pipeline requires torch")
pytest.importorskip("peft", reason="training pipeline requires peft")
pytest.importorskip("lm_eval", reason="evaluation step requires lm-eval-harness")

from main import (
    LOGGER,
    STORAGE_DIR,
    Position,
    WorkflowDefinition,
    WorkflowEdge,
    WorkflowExecutor,
    WorkflowNode,
)


@pytest.mark.slow
def test_end_to_end_training_pipeline():
    run_id = f"test-run-{uuid4().hex[:8]}"
    nodes = [
        WorkflowNode(
            id="dataset",
            type="dataset_build",
            position=Position(x=0, y=0),
            data={"records": [{"text": "hello world"}, {"text": "hello there"}]},
        ),
        WorkflowNode(
            id="tokenize",
            type="tokenize",
            position=Position(x=1, y=0),
            data={},
        ),
        WorkflowNode(
            id="train",
            type="train_sft_lora",
            position=Position(x=2, y=0),
            data={"epochs": 1, "batchSize": 2, "hiddenSize": 16},
        ),
        WorkflowNode(
            id="merge",
            type="merge_lora",
            position=Position(x=3, y=0),
            data={},
        ),
        WorkflowNode(
            id="quantize",
            type="quantize_export",
            position=Position(x=4, y=0),
            data={},
        ),
        WorkflowNode(
            id="eval",
            type="eval_lmeval",
            position=Position(x=5, y=0),
            data={"evalId": "integration"},
        ),
        WorkflowNode(
            id="publish",
            type="registry_publish",
            position=Position(x=6, y=0),
            data={"modelId": "test-model"},
        ),
    ]
    edges = [
        WorkflowEdge(id="e1", fromNode="dataset", fromPort="out", toNode="tokenize", toPort="in"),
        WorkflowEdge(id="e2", fromNode="tokenize", fromPort="out", toNode="train", toPort="in"),
        WorkflowEdge(id="e3", fromNode="train", fromPort="out", toNode="merge", toPort="in"),
        WorkflowEdge(id="e4", fromNode="merge", fromPort="out", toNode="quantize", toPort="in"),
        WorkflowEdge(id="e5", fromNode="quantize", fromPort="out", toNode="eval", toPort="in"),
        WorkflowEdge(id="e6", fromNode="eval", fromPort="out", toNode="publish", toPort="in"),
    ]
    definition = WorkflowDefinition(nodes=nodes, edges=edges)
    executor = WorkflowExecutor(definition, logger=LOGGER, run_id=run_id)
    response = executor.run()

    status_map = {node.id: node.status for node in response.nodes}
    assert all(status == "done" for status in status_map.values())

    quant_meta = executor.node_outputs["quantize"]["quantizedModel"]
    quant_dir = Path(quant_meta["quantizedModelPath"])
    assert (quant_dir / "quantized.pt").exists()

    eval_report = executor.node_outputs["eval"]["evalReport"]
    assert "metrics" in eval_report
    assert eval_report["metrics"]["samples"] >= 1

    registry_path = STORAGE_DIR / "models" / "registry.json"
    assert registry_path.exists()
    registry = json.loads(registry_path.read_text())
    assert "test-model" in registry
    assert registry["test-model"]["quantizedModel"]["quantizedModelPath"] == quant_meta["quantizedModelPath"]


def test_tokenize_without_dataset_marks_failure():
    nodes = [
        WorkflowNode(
            id="tokenize",
            type="tokenize",
            position=Position(x=0, y=0),
            data={},
        )
    ]
    definition = WorkflowDefinition(nodes=nodes, edges=[])
    executor = WorkflowExecutor(definition, logger=LOGGER, run_id="missing-dataset")
    response = executor.run()
    node_state = response.nodes[0]
    assert node_state.status == "failed"
    assert "requires" in node_state.data["error"]
