"""Training pipeline primitives used by the workflow executor.

The real implementation keeps the heavy lifting inside a dedicated module so
it can be unit tested in isolation.  Each step returns a tuple consisting of
outputs that are propagated to downstream nodes and a serialisable state
snapshot for persistence.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

try:  # Optional import, tests exercise CPU fallback.
    import bitsandbytes as bnb  # type: ignore
except Exception:  # pragma: no cover - dependency missing at runtime
    bnb = None  # type: ignore

try:  # pragma: no cover - gracefully handle absence in minimal envs
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover - runtime fallback used in tests
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = object  # type: ignore
    Dataset = object  # type: ignore

try:  # pragma: no cover - dependency is exercised in integration tests
    from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore
except Exception:  # pragma: no cover - fallback for environments without peft
    LoraConfig = None  # type: ignore
    PeftModel = None  # type: ignore

try:  # pragma: no cover - optional dependency for evaluation
    from lm_eval import evaluator as lm_evaluator  # type: ignore
    from lm_eval.api.model import Model as LMEvalModel  # type: ignore
    from lm_eval.api.registry import register_model  # type: ignore
except Exception:  # pragma: no cover - evaluation degrades gracefully
    lm_evaluator = None  # type: ignore
    LMEvalModel = object  # type: ignore


class PipelineError(RuntimeError):
    """Domain specific exception for pipeline failures."""


@dataclass
class PipelineContext:
    run_id: str
    storage_dir: Path
    logger: logging.Logger

    @property
    def run_dir(self) -> Path:
        path = self.storage_dir / "runs" / self.run_id
        path.mkdir(parents=True, exist_ok=True)
        return path


class SimpleTokenizer:
    """Whitespace tokenizer that builds a tiny vocabulary on the fly."""

    def __init__(self) -> None:
        self.token_to_id: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
        self.id_to_token: List[str] = ["<pad>", "<unk>"]

    def encode(self, text: str) -> List[int]:
        tokens: List[int] = []
        for piece in text.lower().split():
            if piece not in self.token_to_id:
                self.token_to_id[piece] = len(self.id_to_token)
                self.id_to_token.append(piece)
            tokens.append(self.token_to_id[piece])
        return tokens

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"token_to_id": self.token_to_id}))

    @classmethod
    def load(cls, path: Path) -> "SimpleTokenizer":
        data = json.loads(path.read_text())
        tokenizer = cls()
        tokenizer.token_to_id = dict(data["token_to_id"])
        tokenizer.id_to_token = [None] * len(tokenizer.token_to_id)
        for token, idx in tokenizer.token_to_id.items():
            if idx >= len(tokenizer.id_to_token):
                tokenizer.id_to_token.extend([None] * (idx - len(tokenizer.id_to_token) + 1))
            tokenizer.id_to_token[idx] = token
        tokenizer.id_to_token = [tok or "<pad>" for tok in tokenizer.id_to_token]
        return tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)


if torch is not None:

    class TokenisedDataset(Dataset):
        """Small dataset turning token sequences into language-modelling targets."""

        def __init__(self, sequences: Iterable[List[int]], context_length: int = 8) -> None:
            self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for sequence in sequences:
                if len(sequence) < 2:
                    continue
                for cursor in range(1, len(sequence)):
                    context = sequence[max(0, cursor - context_length):cursor]
                    if len(context) < context_length:
                        context = [0] * (context_length - len(context)) + list(context)
                    label = sequence[cursor]
                    self.samples.append(
                        (
                            torch.tensor(context, dtype=torch.long),
                            torch.tensor(label, dtype=torch.long),
                        )
                    )
            if not self.samples:
                raise PipelineError("Tokenised dataset does not contain enough data for training.")

        def __len__(self) -> int:  # pragma: no cover - trivial
            return len(self.samples)

        def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover - trivial
            return self.samples[index]


    class TinyCausalLM(nn.Module):
        """Very small causal LM that still supports PEFT adapters."""

        def __init__(self, vocab_size: int, hidden_size: int = 32) -> None:
            super().__init__()
            self.config = SimpleNamespace(task_type="CAUSAL_LM")
            self.embed = nn.Embedding(vocab_size, hidden_size)
            self.hidden_size = hidden_size
            if bnb is not None and hasattr(bnb, "nn") and hasattr(bnb.nn, "Linear4bit"):
                self.decoder = bnb.nn.Linear4bit(hidden_size, vocab_size, bias=False)
                self.uses_4bit = True
            else:
                self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
                self.uses_4bit = False
            self.loss_fn = nn.CrossEntropyLoss()

        def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> SimpleNamespace:
            hidden = self.embed(input_ids)
            pooled = hidden.mean(dim=1)
            logits = self.decoder(pooled)
            loss = None
            if labels is not None:
                loss = self.loss_fn(logits, labels)
            return SimpleNamespace(logits=logits, loss=loss)

        def get_input_embeddings(self) -> nn.Module:  # pragma: no cover - accessor
            return self.embed

        def set_input_embeddings(self, embeddings: nn.Module) -> None:  # pragma: no cover - accessor
            self.embed = embeddings

        def to_dict(self) -> Dict[str, Any]:
            return {"vocab_size": self.embed.num_embeddings, "hidden_size": self.hidden_size}

else:

    class TokenisedDataset:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise PipelineError("PyTorch is required for training but is not available.")


    class TinyCausalLM:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise PipelineError("PyTorch is required for training but is not available.")


def _ensure_torch_available() -> None:
    if torch is None:
        raise PipelineError(
            "PyTorch is required for this operation. Install torch to continue."
        )
    if LoraConfig is None:
        raise PipelineError(
            "PEFT is required for LoRA fine-tuning but is not available. Install peft to continue."
        )


def _ensure_lm_eval_available() -> None:
    if lm_evaluator is None:
        raise PipelineError(
            "lm-eval-harness is required for evaluation but is not installed."
        )


class TrainingPipeline:
    """Encapsulates the stateful pieces of the training workflow."""

    def __init__(self, context: PipelineContext) -> None:
        self.context = context
        self.logger = context.logger
        self.datasets_dir = context.run_dir / "datasets"
        self.tokenized_dir = context.run_dir / "tokenized"
        self.models_dir = context.run_dir / "models"
        self.eval_dir = context.run_dir / "eval"
        for directory in (
            self.datasets_dir,
            self.tokenized_dir,
            self.models_dir,
            self.eval_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset build
    # ------------------------------------------------------------------
    def dataset_build(self, node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        dataset_records = list(node_data.get("records") or inputs.get("records") or [])
        text_blob = node_data.get("text") or inputs.get("text")
        if text_blob and not dataset_records:
            dataset_records = [{"text": text_blob}]

        if not dataset_records:
            raise PipelineError("Dataset builder requires at least one record.")

        dataset_id = node_data.get("datasetId") or f"dataset-{int(time.time())}"
        dataset_dir = self.datasets_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        data_path = dataset_dir / "dataset.jsonl"
        with data_path.open("w", encoding="utf-8") as handle:
            for record in dataset_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        metadata = {
            "datasetId": dataset_id,
            "path": str(data_path),
            "records": len(dataset_records),
            "updatedAt": int(time.time()),
        }
        state = {**node_data, **metadata}
        self.logger.info("Dataset %s built with %d records", dataset_id, metadata["records"])
        return {"dataset": metadata, "out": metadata}, state

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------
    def tokenize(self, node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        dataset = inputs.get("dataset") or node_data.get("dataset")
        if not dataset:
            raise PipelineError("Tokeniser requires dataset metadata from the previous step.")
        dataset_path = Path(dataset["path"])
        tokenizer = SimpleTokenizer()
        sequences: List[List[int]] = []
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                text = str(payload.get("text") or payload)
                sequences.append(tokenizer.encode(text))
        if not sequences:
            raise PipelineError("Dataset is empty and cannot be tokenised.")

        token_dir = self.tokenized_dir / dataset["datasetId"]
        token_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_path = token_dir / "tokenizer.json"
        tokenizer.save(tokenizer_path)
        tokens_path = token_dir / "tokens.json"
        tokens_path.write_text(json.dumps(sequences))
        metadata = {
            "datasetId": dataset["datasetId"],
            "tokenizer": str(tokenizer_path),
            "tokens": str(tokens_path),
            "sequenceCount": len(sequences),
            "vocabSize": tokenizer.vocab_size,
        }
        state = {**node_data, **metadata}
        self.logger.info(
            "Dataset %s tokenised into %d sequences", dataset["datasetId"], metadata["sequenceCount"]
        )
        return {"tokenizedDataset": metadata, "out": metadata}, state

    # ------------------------------------------------------------------
    # QLoRA training
    # ------------------------------------------------------------------
    def train_sft_lora(self, node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        _ensure_torch_available()
        tokenized = inputs.get("tokenizedDataset") or node_data.get("tokenizedDataset")
        if not tokenized:
            raise PipelineError("LoRA training requires a tokenised dataset.")
        token_file = Path(tokenized["tokens"])
        tokenizer_path = Path(tokenized["tokenizer"])
        sequences: List[List[int]] = json.loads(token_file.read_text())
        tokenizer = SimpleTokenizer.load(tokenizer_path)
        dataset = TokenisedDataset(sequences, context_length=int(node_data.get("contextLength", 8)))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TinyCausalLM(vocab_size=tokenizer.vocab_size, hidden_size=int(node_data.get("hiddenSize", 32)))
        model.to(device)
        lora_config = LoraConfig(
            r=int(node_data.get("loraRank", 8)),
            lora_alpha=int(node_data.get("loraAlpha", 16)),
            lora_dropout=float(node_data.get("loraDropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["decoder"],
        )
        peft_model = get_peft_model(model, lora_config)
        peft_model.to(device)
        dataloader = DataLoader(dataset, batch_size=int(node_data.get("batchSize", 2)), shuffle=True)
        optimiser = torch.optim.AdamW(peft_model.parameters(), lr=float(node_data.get("learningRate", 1e-3)))
        epochs = int(node_data.get("epochs", 1))
        peft_model.train()
        final_loss = None
        for _ in range(epochs):
            for batch_inputs, batch_labels in dataloader:
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)
                optimiser.zero_grad()
                outputs = peft_model(batch_inputs, labels=batch_labels)
                loss = outputs.loss
                if loss is None:
                    raise PipelineError("Model did not return a loss during training.")
                loss.backward()
                optimiser.step()
                final_loss = float(loss.detach().cpu().item())
        if final_loss is None:
            raise PipelineError("Training loop did not run; dataset may be empty.")

        model_dir = self.models_dir / node_data.get("modelId", "qlora")
        base_dir = model_dir / "base"
        adapter_dir = model_dir / "adapter"
        base_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), base_dir / "model.pt")
        (base_dir / "config.json").write_text(json.dumps(model.to_dict()))
        (base_dir / "tokenizer.json").write_text(tokenizer_path.read_text())
        peft_model.save_pretrained(adapter_dir)
        metadata = {
            "baseModelPath": str(base_dir),
            "adapterPath": str(adapter_dir),
            "loss": final_loss,
            "device": str(device),
            "updatedAt": int(time.time()),
        }
        state = {**node_data, **metadata}
        self.logger.info(
            "QLoRA training finished on %s with final loss %.4f", metadata["device"], final_loss
        )
        return {"loraWeights": metadata, "out": metadata}, state

    # ------------------------------------------------------------------
    # Merge LoRA adapters
    # ------------------------------------------------------------------
    def merge_lora(self, node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        _ensure_torch_available()
        lora_meta = inputs.get("loraWeights") or node_data.get("loraWeights")
        if not lora_meta:
            raise PipelineError("Merge step requires LoRA training outputs.")
        base_dir = Path(lora_meta["baseModelPath"])
        adapter_dir = Path(lora_meta["adapterPath"])
        config = json.loads((base_dir / "config.json").read_text())
        tokenizer_path = base_dir / "tokenizer.json"
        model = TinyCausalLM(vocab_size=config["vocab_size"], hidden_size=config["hidden_size"])
        state_dict = torch.load(base_dir / "model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        peft_model = PeftModel.from_pretrained(model, adapter_dir)
        peft_model = peft_model.merge_and_unload()
        merged_dir = Path(lora_meta.get("mergedModelPath") or base_dir.parent / "merged")
        merged_dir.mkdir(parents=True, exist_ok=True)
        torch.save(peft_model.state_dict(), merged_dir / "model.pt")
        (merged_dir / "config.json").write_text(json.dumps(config))
        (merged_dir / "tokenizer.json").write_text(tokenizer_path.read_text())
        metadata = {
            "mergedModelPath": str(merged_dir),
            "baseModelPath": str(base_dir),
            "adapterPath": str(adapter_dir),
            "updatedAt": int(time.time()),
        }
        state = {**node_data, **metadata}
        self.logger.info("Merged LoRA adapters into base model at %s", merged_dir)
        return {"mergedModel": metadata, "model": metadata, "out": metadata}, state

    # ------------------------------------------------------------------
    # Quantisation and export
    # ------------------------------------------------------------------
    def quantize_export(self, node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        _ensure_torch_available()
        merged_meta = inputs.get("mergedModel") or node_data.get("mergedModel")
        if not merged_meta:
            raise PipelineError("Quantisation requires a merged model output.")
        merged_dir = Path(merged_meta["mergedModelPath"])
        config = json.loads((merged_dir / "config.json").read_text())
        state_dict = torch.load(merged_dir / "model.pt", map_location="cpu")
        quantized_state: Dict[str, Any] = {}
        for name, tensor in state_dict.items():
            tensor = tensor.to(torch.float32)
            if bnb is not None:
                quantised, quant_state = bnb.functional.quantize_4bit(tensor)
                quantized_state[name] = {
                    "dtype": "nf4",
                    "quantised": quantised.cpu(),
                    "quant_state": quant_state,
                }
            else:
                quantized_state[name] = {
                    "dtype": "int8",
                    "quantised": tensor.to(torch.int8),
                    "scale": float(tensor.abs().max().cpu().item() or 1.0),
                }
        quant_dir = Path(node_data.get("exportPath") or merged_dir.parent / "quantized")
        quant_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"config": config, "state": quantized_state}, quant_dir / "quantized.pt")
        (quant_dir / "tokenizer.json").write_text((merged_dir / "tokenizer.json").read_text())
        metadata = {
            "quantizedModelPath": str(quant_dir),
            "format": "nf4" if bnb is not None else "int8",
            "updatedAt": int(time.time()),
        }
        state = {**node_data, **metadata}
        self.logger.info("Quantised model exported to %s", quant_dir)
        return {"quantizedModel": metadata, "model": metadata, "out": metadata}, state

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def eval_lmeval(self, node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        _ensure_lm_eval_available()
        _ensure_torch_available()
        quant_meta = inputs.get("quantizedModel") or node_data.get("quantizedModel")
        if not quant_meta:
            raise PipelineError("Evaluation requires the quantised model output.")
        quant_dir = Path(quant_meta["quantizedModelPath"])
        bundle = torch.load(quant_dir / "quantized.pt", map_location="cpu")
        config = bundle["config"]
        quant_state = bundle["state"]
        tokenizer = SimpleTokenizer.load(quant_dir / "tokenizer.json")

        class TinyQuantisedLM(LMEvalModel):
            def __init__(self) -> None:
                super().__init__()
                self.config = {"model_id": "tiny-quantised"}
                self.device = "cpu"
                self.model = TinyCausalLM(vocab_size=config["vocab_size"], hidden_size=config["hidden_size"])
                restored = {}
                for name, payload in quant_state.items():
                    if payload["dtype"] == "nf4" and bnb is not None:
                        restored[name] = bnb.functional.dequantize_4bit(
                            payload["quantised"].to(torch.device("cpu")), payload["quant_state"]
                        )
                    else:
                        scale = payload.get("scale", 1.0)
                        restored[name] = payload["quantised"].to(torch.float32) / (scale or 1.0)
                self.model.load_state_dict(restored)

            # lm_eval expects these attributes
            def loglikelihood(self, requests):  # type: ignore[override]
                results = []
                for context, continuation in requests:
                    ctx_tokens = tokenizer.encode(context)
                    cont_tokens = tokenizer.encode(continuation)
                    if not cont_tokens:
                        results.append((0.0, True))
                        continue
                    input_ids = torch.tensor(
                        [ctx_tokens[-8:]], dtype=torch.long
                    )  # context window matches training
                    logits = self.model(input_ids).logits
                    log_probs = torch.log_softmax(logits, dim=-1)
                    score = 0.0
                    for token in cont_tokens:
                        score += float(log_probs[0, token].item())
                    results.append((score, True))
                return results

            def loglikelihood_rolling(self, requests):  # type: ignore[override]
                return self.loglikelihood(requests)

            def greedy_until(self, requests):  # type: ignore[override]
                generations = []
                for context, until in requests:
                    tokens = tokenizer.encode(context)
                    input_ids = torch.tensor([tokens[-8:]], dtype=torch.long)
                    logits = self.model(input_ids).logits
                    next_token = int(torch.argmax(logits, dim=-1)[0].item())
                    generated = tokenizer.id_to_token[next_token]
                    generations.append((generated, None))
                return generations

            @property
            def eot_token_id(self):  # pragma: no cover - metadata access
                return 0

            @property
            def max_gen_toks(self):  # pragma: no cover - metadata access
                return 16

            @property
            def max_length(self):  # pragma: no cover - metadata access
                return 8

            @property
            def requires_attention_mask(self):  # pragma: no cover - metadata access
                return False

        register_model("tiny-quantised", TinyQuantisedLM)
        eval_dataset = {
            "task": "simple-perplexity",
            "dataset": [
                {"context": " ", "continuation": "hello"},
                {"context": "hello", "continuation": "world"},
            ],
        }

        def _simple_task_evaluator(model: TinyQuantisedLM):
            requests = [(sample["context"], sample["continuation"]) for sample in eval_dataset["dataset"]]
            scores = model.loglikelihood(requests)
            perplexity = float(torch.exp(-torch.tensor([score for score, _ in scores]).mean()).item())
            return {"perplexity": perplexity, "samples": len(scores)}

        model = TinyQuantisedLM()
        metrics = _simple_task_evaluator(model)
        result_path = self.eval_dir / f"{node_data.get('evalId', 'eval')}.json"
        result_path.write_text(json.dumps(metrics))
        metadata = {
            "metrics": metrics,
            "resultPath": str(result_path),
            "updatedAt": int(time.time()),
        }
        state = {**node_data, **metadata}
        self.logger.info("Evaluation complete with metrics: %s", metrics)
        return {"evalReport": metadata, "out": metadata}, state

    # ------------------------------------------------------------------
    # Registry publish
    # ------------------------------------------------------------------
    def registry_publish(self, node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        quant_meta = (
            inputs.get("quantizedModel")
            or inputs.get("model")
            or node_data.get("quantizedModel")
        )
        eval_meta = inputs.get("evalReport") or node_data.get("evalReport")
        if not quant_meta:
            raise PipelineError("Registry publish step requires a quantised model artifact.")
        model_id = node_data.get("modelId") or f"model-{self.context.run_id}"
        registry_dir = self.context.storage_dir / "models"
        registry_dir.mkdir(parents=True, exist_ok=True)
        registry_path = registry_dir / "registry.json"
        if registry_path.exists():
            registry = json.loads(registry_path.read_text())
        else:
            registry = {}
        entry = {
            "modelId": model_id,
            "quantizedModel": quant_meta,
            "evaluation": eval_meta,
            "publishedAt": int(time.time()),
            "runId": self.context.run_id,
        }
        registry[model_id] = entry
        registry_path.write_text(json.dumps(registry, indent=2))
        state = {**node_data, **entry}
        self.logger.info("Published model %s to registry", model_id)
        return {"model": entry, "out": entry}, state
