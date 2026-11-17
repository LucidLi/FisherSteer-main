import argparse
import hashlib
import inspect
import json
import random
import sys
from time import perf_counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.fisher import fit_fisher, rank_fisher_layers
from utils.hidden_states import capture_decoder_outputs

from evaluate_asr import (build_inputs as shared_build_inputs,
                          load_model_and_tokenizer_for_online as shared_load_model,
                          is_prompt_optimized_attack, uses_fastchat_template)


EPS = 1e-12
LABEL_ORDER = ("J", "B", "H")
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABEL_ORDER)}
MODEL_RELATIVE_DEFAULTS = {
    "vicuna": Path("models") / "vicuna-7b-v1.5",
    "vicuna-7b": Path("models") / "vicuna-7b-v1.5",
    "vicuna-7b-v1.5": Path("models") / "vicuna-7b-v1.5",
    "vicuna-13b": Path("models") / "vicuna-13b-v1.5",
    "vicuna-13b-v1.5": Path("models") / "vicuna-13b-v1.5",
    "llama2": Path("models") / "Llama-2-7b-chat-hf",
    "llama-2": Path("models") / "Llama-2-7b-chat-hf",
    "llama-2-7b-chat-hf": Path("models") / "Llama-2-7b-chat-hf",
    "mistral": Path("models") / "Mistral-7B-Instruct-v0.2",
    "mistral-7b": Path("models") / "Mistral-7B-Instruct-v0.2",
    "mistral-7b-instruct-v0.2": Path("models") / "Mistral-7B-Instruct-v0.2",
    "llama3": Path("models") / "Meta-Llama-3-8B-Instruct",
    "llama-3": Path("models") / "Meta-Llama-3-8B-Instruct",
    "llama-3-8b": Path("models") / "Meta-Llama-3-8B-Instruct",
    "llama-3-8b-instruct": Path("models") / "Meta-Llama-3-8B-Instruct",
    "meta-llama-3-8b-instruct": Path("models") / "Meta-Llama-3-8B-Instruct",
}
TRAIN_DATASET_RELATIVE_DEFAULTS = {
    "vicuna": Path("data") / "train" / "comprehensive_vicuna-7b_train.json",
    "vicuna-7b": Path("data") / "train" / "comprehensive_vicuna-7b_train.json",
    "vicuna-7b-v1.5": Path("data") / "train" / "comprehensive_vicuna-7b_train.json",
    "vicuna-13b": Path("data") / "train" / "comprehensive_vicuna-13b_train.json",
    "vicuna-13b-v1.5": Path("data") / "train" / "comprehensive_vicuna-13b_train.json",
    "llama2": Path("data") / "train" / "comprehensive_llama-2_train.json",
    "llama-2": Path("data") / "train" / "comprehensive_llama-2_train.json",
    "llama-2-7b-chat-hf": Path("data") / "train" / "comprehensive_llama-2_train.json",
    "mistral": Path("data") / "train" / "comprehensive_mistral_train.json",
    "mistral-7b": Path("data") / "train" / "comprehensive_mistral_train.json",
    "mistral-7b-instruct-v0.2": Path("data") / "train" / "comprehensive_mistral_train.json",
    "llama3": Path("data") / "train" / "comprehensive_llama-3_train.json",
    "llama-3": Path("data") / "train" / "comprehensive_llama-3_train.json",
    "llama-3-8b": Path("data") / "train" / "comprehensive_llama-3_train.json",
    "llama-3-8b-instruct": Path("data") / "train" / "comprehensive_llama-3_train.json",
    "meta-llama-3-8b-instruct": Path("data") / "train" / "comprehensive_llama-3_train.json",
}
EXCLUDED_TRAIN_METHODS = {"base64", "zulu"}


from utils.detector import PrefillMLPDetector


LABEL_TO_FIELD = {
    "J": "J_prompt",
    "B": "B_prompt",
    "H": "H_prompt",
}


def available_labels_for_sample(sample: dict):
    labels = []
    for label in LABEL_ORDER:
        field = LABEL_TO_FIELD[label]
        if field in sample and sample[field]:
            labels.append(label)
    return labels


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_name(name: str):
    text = str(name).strip()
    text = "".join(ch for ch in text if ch.isalnum() or ch in ("_", "-"))
    return text or "unknown"


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def iter_progress(total: int, desc: str):
    if tqdm is None:
        return None
    return tqdm(total=total, desc=desc, dynamic_ncols=True)


def load_samples(dataset_path: Path):
    dataset_path = resolve_local_path(dataset_path)
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"Expected a list or a dict with key 'data': {dataset_path}")
    retained = [
        sample
        for sample in data
        if str(sample.get("method", "")).strip().lower() not in EXCLUDED_TRAIN_METHODS
    ]
    excluded_count = len(data) - len(retained)
    if excluded_count:
        print(
            f"[Info] Excluded {excluded_count} Base64/Zulu rows from PRI detector training.",
            flush=True,
        )
    return retained


def deduplicate_prompt_fields(samples):
    """Train takes precedence; each exact input contributes once per class."""
    seen = set()
    kept = []
    for sample in sorted(samples, key=lambda row: str(row.get("split", "")).lower() not in {"train", ""}):
        row = dict(sample)
        for label, field in LABEL_TO_FIELD.items():
            text = row.get(field)
            if not text:
                continue
            key = (label, text)
            if key in seen:
                row.pop(field)
            else:
                seen.add(key)
        if available_labels_for_sample(row):
            kept.append(row)
    return kept


def prompt_counts(samples):
    from collections import Counter
    return dict(Counter(f"{str(s.get('split') or 'unspecified')}:{c}"
                        for s in samples for c in available_labels_for_sample(s)))


def infer_template_name(model_name: str):
    name = str(model_name).lower()
    if name in {"llama2", "llama-2", "llama-2-7b-chat", "llama-2-7b-chat-hf"}:
        return "llama-2"
    if name in {"llama3", "llama-3", "llama-3-8b", "llama-3-8b-instruct", "meta-llama-3-8b-instruct"}:
        return "llama-3"
    if name in {"mistral", "mistral-7b", "mistral-7b-instruct", "mistral-7b-instruct-v0.2"}:
        return "mistral"
    if name in {"vicuna", "vicuna-7b", "vicuna-13b", "vicuna-7b-v1.5", "vicuna-13b-v1.5"}:
        return "vicuna"
    if name.startswith("qwen"):
        return "qwen"
    return name


def resolve_local_path(path: Path):
    path = Path(path)
    if path.is_absolute():
        return path
    candidates = [
        (ROOT_DIR / path).resolve(),
        (Path.cwd() / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_model_path(model_name: str, model_path: Optional[Path] = None):
    if model_path is not None:
        return resolve_local_path(model_path)
    name = str(model_name).lower()
    if name in MODEL_RELATIVE_DEFAULTS:
        default_path = resolve_local_path(MODEL_RELATIVE_DEFAULTS[name])
        if default_path.exists():
            return default_path
        models_dir = ROOT_DIR / "models"
        if models_dir.exists():
            accepted_names = {default_path.name.lower()}
            if name in {"llama3", "llama-3", "llama-3-8b", "llama-3-8b-instruct", "meta-llama-3-8b-instruct"}:
                accepted_names.add("llama-3-8b-instruct")
            for candidate in models_dir.iterdir():
                if candidate.is_dir() and candidate.name.lower() in accepted_names:
                    return candidate.resolve()
        return default_path
    return resolve_local_path(Path("models") / model_name)


def default_device():
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        return "auto"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def build_inputs(tokenizer, template_name, prompt, device, method=""):
    return shared_build_inputs(tokenizer, template_name, prompt, device,
                               whitebox_attacker=is_prompt_optimized_attack(method),
                               use_fastchat_template=uses_fastchat_template(method))


def filter_model_inputs_online(model, model_inputs: dict):
    accepted = set(inspect.signature(model.forward).parameters.keys())
    return {key: value for key, value in model_inputs.items() if key in accepted}


def get_input_device(model):
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def load_model_and_tokenizer_for_online(model_path, device, gpu_memory, cpu_memory):
    return shared_load_model(model_path, device, gpu_memory, cpu_memory)


def usable_hidden_states(outputs):
    hidden_states = list(outputs.hidden_states)
    if len(hidden_states) > 1:
        return hidden_states[1:]
    return hidden_states


def stack_prefill_hidden_states(outputs, num_tokens: int):
    hidden_states = usable_hidden_states(outputs)
    seq_len = hidden_states[0].shape[1]
    if seq_len < num_tokens:
        raise ValueError(f"Prompt length {seq_len} is shorter than prefill_num_tokens={num_tokens}.")
    matrices = []
    for token_index in range(1, num_tokens + 1):
        position = seq_len - token_index
        matrices.append(
            torch.stack(
                [layer_output[0, position, :].detach().float().cpu() for layer_output in hidden_states],
                dim=0,
            )
        )
    return matrices


def parse_args():
    parser = argparse.ArgumentParser("Train the PRI prefill detector and select critical layers.")
    # vicuna-7b、llama-2、mistral、llama-3、vicuna-13b
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Training JSON. Defaults to data/train/comprehensive_<model>_train.json.",
    )
    parser.add_argument("--dataset-tag", type=str, default=None)
    parser.add_argument("--train-attack", default=None, help="Keep one J attack plus all B/H prompts for cross-attack transfer experiments.")
    parser.add_argument("--model-name", type=str, default="mistral")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--train-size", type=int, default=-1)
    parser.add_argument("--prefill-num-tokens", type=int, default=10)
    parser.add_argument("--td-topk-m-select", type=int, default=None)
    parser.add_argument("--td-topk-max-k", type=int, default=10)
    parser.add_argument("--artifact-root", type=Path, default=ROOT_DIR / "training_results")
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--auto-gpu-memory", type=str, default="22GiB")
    parser.add_argument("--auto-cpu-memory", type=str, default="64GiB")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--mlp-encoder-out-dim", type=int, default=128)
    parser.add_argument("--mlp-dropout", type=float, default=0.2)
    parser.add_argument("--mlp-epochs", type=int, default=80)
    parser.add_argument("--mlp-batch-size", type=int, default=16)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--fisher-rho", type=float, default=0.1, help="Regularization strength rho (paper default: 0.1).")
    parser.add_argument("--fisher-variance-floor", type=float, default=1e-12)
    parser.add_argument("--fisher-device", type=str, default="cpu")
    args = parser.parse_args()
    if args.td_topk_m_select is None:
        args.td_topk_m_select = args.prefill_num_tokens
    if args.prefill_num_tokens <= 0 or args.td_topk_m_select != args.prefill_num_tokens:
        parser.error("Paper mode requires the same positive m for detection and Fisher layer scoring.")
    if args.td_topk_max_k <= 0:
        parser.error("--td-topk-max-k must be positive.")
    if not np.isfinite(args.fisher_rho) or args.fisher_rho <= 0:
        parser.error("--fisher-rho must be positive and finite.")
    return args


def normalize_last_dim(x: np.ndarray):
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, EPS)


def batch_arrays_from_examples(examples: List[dict], indices):
    x = np.stack(
        [
            normalize_last_dim(examples[int(idx)]["layer_matrix_raw"].astype(np.float32, copy=False))
            for idx in indices
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    y = np.asarray([examples[int(idx)]["y"] for idx in indices], dtype=np.int64)
    return x, y


def example_label_array(examples: List[dict]):
    return np.asarray([example["y"] for example in examples], dtype=np.int64)


def build_artifact_dir(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_aliases = {
        "mistral-7b": "mistral",
        "mistral-7b-instruct-v0.2": "mistral",
        "llama2": "llama-2",
        "llama-2-7b-chat-hf": "llama-2",
        "llama3": "llama-3",
        "llama-3-8b": "llama-3",
        "llama-3-8b-instruct": "llama-3",
        "meta-llama-3-8b-instruct": "llama-3",
        "vicuna-7b-v1.5": "vicuna-7b",
        "vicuna-13b-v1.5": "vicuna-13b",
    }
    canonical_model_name = model_aliases.get(str(args.model_name).strip().lower(), args.model_name)
    model_name = sanitize_name(canonical_model_name)
    dataset_tag = sanitize_name(args.dataset_tag)
    run_name = (
        f"{dataset_tag}_prefill{args.prefill_num_tokens}_train{args.train_size}_"
        f"m{args.td_topk_m_select}_k{args.td_topk_max_k}_seed{args.seed}_{timestamp}"
    )
    return args.artifact_root / model_name / run_name


@torch.inference_mode()
def extract_one_prompt(model, tokenizer, template_name: str, prompt: str, device: str, num_tokens: int, method=""):
    inputs = build_inputs(tokenizer, template_name, prompt, device, method)
    forward_kwargs = {
        **inputs,
        "output_hidden_states": True,
        "return_dict": True,
        "use_cache": False,
    }
    filtered = filter_model_inputs_online(model, forward_kwargs)
    with capture_decoder_outputs(model) as raw_states:
        outputs = model(**filtered)
    outputs.hidden_states = (None, *raw_states)
    return stack_prefill_hidden_states(outputs, num_tokens)


def prepare_training_examples(
    model,
    tokenizer,
    samples: List[dict],
    template_name: str,
    device: str,
    num_tokens: int,
    progress_every: int,
):
    examples = []
    sample_labels = [available_labels_for_sample(sample) for sample in samples]
    missing = [
        sample.get("id", sample_idx)
        for sample_idx, (sample, labels) in enumerate(zip(samples, sample_labels))
        if not labels
    ]
    if missing:
        raise ValueError(f"Samples missing all J/B/H prompt fields: {missing[:10]}")
    total_forwards = sum(len(labels) for labels in sample_labels)
    progress = iter_progress(total_forwards, "extract available J/B/H prefill states")
    completed = 0
    for sample_idx, sample in enumerate(samples):
        if progress is None and (sample_idx == 0 or (sample_idx + 1) % max(int(progress_every), 1) == 0):
            print(
                "[Info] hidden-state extraction "
                f"sample={sample_idx + 1}/{len(samples)} "
                f"split={sample.get('split')} method={sample.get('method')} id={sample.get('id')}",
                flush=True,
            )
        for label in sample_labels[sample_idx]:
            field = LABEL_TO_FIELD[label]
            if progress is not None:
                progress.set_postfix(
                    {
                        "sample": f"{sample_idx + 1}/{len(samples)}",
                        "split": sample.get("split", ""),
                        "method": str(sample.get("method", ""))[:12],
                        "label": label,
                    }
                )
            prompt_text = sample[field]
            approx_prompt_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
            if progress is not None:
                progress.set_postfix(
                    {
                        "sample": f"{sample_idx + 1}/{len(samples)}",
                        "split": sample.get("split", ""),
                        "method": str(sample.get("method", ""))[:12],
                        "label": label,
                        "tok": approx_prompt_tokens,
                    }
                )
            start = perf_counter()
            matrices = extract_one_prompt(model, tokenizer, template_name, prompt_text, device, num_tokens, sample.get("method", "") if label == "J" else "")
            elapsed = perf_counter() - start
            completed += 1
            if progress is not None:
                progress.update(1)
            elif completed % max(int(progress_every), 1) == 0:
                print(f"[Info] extracted {completed}/{total_forwards} available J/B/H prompts", flush=True)
            if elapsed >= 30.0:
                print(
                    "[Warn] slow hidden-state forward "
                    f"sample={sample_idx + 1}/{len(samples)} split={sample.get('split')} "
                    f"method={sample.get('method')} id={sample.get('id')} label={label} "
                    f"prompt_tokens={approx_prompt_tokens} elapsed={elapsed:.1f}s",
                    flush=True,
                )
            for token_index, layer_matrix in enumerate(matrices, start=1):
                raw = layer_matrix.numpy().astype(np.float32, copy=False)
                examples.append(
                    {
                        "sample_id": int(sample.get("id", sample_idx)),
                        "prompt_key": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                        "split": sample.get("split"),
                        "method": sample.get("method"),
                        "source": sample.get("source"),
                        "source_id": sample.get("source_id"),
                        "token_index": int(token_index),
                        "label": label,
                        "y": int(LABEL_TO_INDEX[label]),
                        "layer_matrix_raw": raw,
                    }
                )
    if progress is not None:
        progress.close()
    print(
        f"[Info] Finished hidden-state extraction: {len(samples)} samples, "
        f"{total_forwards} prompt forwards, {len(examples)} token examples.",
        flush=True,
    )
    return examples


def split_examples_by_declared_split(examples: List[dict]):
    declared = [str(example.get("split", "")).strip().lower() for example in examples if example.get("split")]
    if not declared:
        return None
    unknown = sorted({split for split in declared if split not in {"train", "val", "validation"}})
    if unknown:
        raise ValueError(f"Unsupported split values in dataset: {unknown}. Expected train/val.")

    train_examples = [example for example in examples if str(example.get("split", "")).strip().lower() == "train"]
    val_examples = [
        example
        for example in examples
        if str(example.get("split", "")).strip().lower() in {"val", "validation"}
    ]
    if not train_examples or not val_examples:
        raise ValueError("Declared split mode requires at least one train example and one val example.")
    if len(train_examples) + len(val_examples) != len(examples):
        raise ValueError("Every example must declare train or val when using declared splits.")
    validate_prompt_disjoint(train_examples, val_examples)
    train_ids = sorted({int(example["sample_id"]) for example in train_examples})
    val_ids = sorted({int(example["sample_id"]) for example in val_examples})
    return train_examples, val_examples, train_ids, val_ids, "declared_split"


def split_examples_by_sample_id(examples: List[dict], val_ratio: float, seed: int):
    ids = sorted({int(example["sample_id"]) for example in examples})
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    val_count = max(1, int(round(len(ids) * val_ratio)))
    val_ids = set(ids[:val_count])
    train_examples = [example for example in examples if int(example["sample_id"]) not in val_ids]
    val_examples = [example for example in examples if int(example["sample_id"]) in val_ids]
    if not train_examples or not val_examples:
        raise ValueError("Need at least two input groups and a nonempty train/validation split.")
    validate_prompt_disjoint(train_examples, val_examples)
    return train_examples, val_examples, sorted(set(ids) - val_ids), sorted(val_ids), "random_by_sample_id"


def evaluate_examples(model, examples: List[dict], device: str, batch_size: int):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    all_preds = []
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        indices = np.arange(len(examples))
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            x_batch, y_batch = batch_arrays_from_examples(examples, batch_idx)
            x_tensor = torch.tensor(x_batch, dtype=torch.float32, device=device)
            y_tensor = torch.tensor(y_batch, dtype=torch.long, device=device)
            logits = model(x_tensor)
            loss = criterion(logits, y_tensor)
            total_loss += float(loss.item())
            total_count += int(len(batch_idx))
            all_preds.append(torch.argmax(logits, dim=-1).detach().cpu().numpy())
    preds = np.concatenate(all_preds, axis=0) if all_preds else np.asarray([], dtype=np.int64)
    avg_loss = total_loss / max(total_count, 1)
    return float(avg_loss), preds


def validation_metrics_from_examples(model, val_examples: List[dict], args):
    y_val = example_label_array(val_examples)
    val_loss, preds = evaluate_examples(model, val_examples, args.device, args.mlp_batch_size)
    overall_acc = float(np.mean(preds == y_val))
    labels = list(LABEL_ORDER)
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for truth, pred in zip(y_val, preds):
        confusion[int(truth), int(pred)] += 1

    acc_by_label = {}
    count_by_label = {}
    for label, label_idx in LABEL_TO_INDEX.items():
        mask = y_val == int(label_idx)
        count = int(mask.sum())
        count_by_label[label] = count
        acc_by_label[label] = float(np.mean(preds[mask] == y_val[mask])) if count else None

    method_values = sorted({str(example.get("method") or "unknown") for example in val_examples})
    acc_by_method = {}
    count_by_method = {}
    for method in method_values:
        indices = [idx for idx, example in enumerate(val_examples) if str(example.get("method") or "unknown") == method]
        count_by_method[method] = int(len(indices))
        if indices:
            acc_by_method[method] = float(np.mean(preds[indices] == y_val[indices]))
        else:
            acc_by_method[method] = None

    return {
        "val_loss": val_loss,
        "overall_val_acc": overall_acc,
        "val_acc_by_label": acc_by_label,
        "val_count_by_label": count_by_label,
        "val_acc_by_method": acc_by_method,
        "val_count_by_method": count_by_method,
        "confusion_matrix_labels": labels,
        "confusion_matrix_rows_true_cols_pred": confusion.tolist(),
    }


def train_detector(examples: List[dict], args, artifact_dir: Path):
    declared_split = split_examples_by_declared_split(examples)
    if declared_split is None:
        train_examples, val_examples, train_ids, val_ids, split_mode = split_examples_by_sample_id(
            examples,
            args.val_ratio,
            args.seed,
        )
    else:
        train_examples, val_examples, train_ids, val_ids, split_mode = declared_split
    if not 0 < args.val_ratio < 1 or args.mlp_epochs <= 0 or args.mlp_batch_size <= 0:
        raise ValueError("Require 0 < val_ratio < 1, positive epochs and batch size.")
    first_shape = train_examples[0]["layer_matrix_raw"].shape

    model = PrefillMLPDetector(
        input_dim=int(first_shape[-1]),
        hidden_dim=args.mlp_hidden_dim,
        encoder_out_dim=args.mlp_encoder_out_dim,
        num_classes=len(LABEL_ORDER),
        dropout=args.mlp_dropout,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.mlp_lr),
        weight_decay=float(args.mlp_weight_decay),
    )
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_val_acc = -1.0
    history = []

    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.mlp_epochs + 1):
        model.train()
        indices = np.arange(len(train_examples))
        rng.shuffle(indices)
        losses = []
        for start in range(0, len(indices), args.mlp_batch_size):
            batch_idx = indices[start : start + args.mlp_batch_size]
            x_batch, y_batch = batch_arrays_from_examples(train_examples, batch_idx)
            batch_x = torch.tensor(x_batch, dtype=torch.float32, device=args.device)
            batch_y = torch.tensor(y_batch, dtype=torch.long, device=args.device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_loss, val_preds = evaluate_examples(model, val_examples, args.device, args.mlp_batch_size)
        y_val = example_label_array(val_examples)
        val_acc = float(np.mean(val_preds == y_val))
        train_loss = float(np.mean(losses)) if losses else 0.0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})
        print(f"[Info] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("MLP training did not produce a best state.")
    model.load_state_dict(best_state)
    validation_metrics = validation_metrics_from_examples(model, val_examples, args)
    weights_path = artifact_dir / "prefill_detector_mlp.pt"
    torch.save({"state_dict": model.state_dict()}, weights_path)
    meta = {
        "weights_path": str(weights_path),
        "label_order": list(LABEL_ORDER),
        "input_dim": int(first_shape[-1]),
        "num_layers": int(first_shape[0]),
        "hidden_dim": int(args.mlp_hidden_dim),
        "encoder_out_dim": int(args.mlp_encoder_out_dim),
        "dropout": float(args.mlp_dropout),
        "num_tokens": int(args.prefill_num_tokens),
        "artifact_format_version": 2,
        "hidden_state_source": "raw_decoder_block_output",
        "training_mode": "prefill_only_joint_across_tokens",
    }
    write_json(artifact_dir / "prefill_detector_meta.json", meta)
    write_json(artifact_dir / "prefill_training_history.json", {"history": history})
    write_json(artifact_dir / "prefill_validation_metrics.json", validation_metrics)
    print(
        "[Info] best validation acc="
        f"{validation_metrics['overall_val_acc']:.4f} | "
        f"J={validation_metrics['val_acc_by_label']['J']:.4f} "
        f"B={validation_metrics['val_acc_by_label']['B']:.4f} "
        f"H={validation_metrics['val_acc_by_label']['H']:.4f}"
    )
    return model, meta, train_ids, val_ids, split_mode, validation_metrics


def compute_centers_raw(examples: List[dict], num_tokens: int, num_layers: int, hidden_dim: int):
    sums = {
        label: np.zeros((num_tokens, num_layers, hidden_dim), dtype=np.float64)
        for label in LABEL_ORDER
    }
    counts = {label: np.zeros((num_tokens, num_layers), dtype=np.int64) for label in LABEL_ORDER}
    for example in examples:
        label = example["label"]
        token_idx = int(example["token_index"]) - 1
        raw = example["layer_matrix_raw"].astype(np.float64, copy=False)
        sums[label][token_idx] += raw
        counts[label][token_idx] += 1
    centers = {}
    for label in LABEL_ORDER:
        centers[label] = sums[label] / np.maximum(counts[label][..., None], 1)
    return centers


def validate_prompt_disjoint(train, val):
    train_keys = {e.get("prompt_key", e["sample_id"]) for e in train}
    val_keys = {e.get("prompt_key", e["sample_id"]) for e in val}
    if train_keys & val_keys:
        raise ValueError("Identical prompts occur in both train and validation; fix input-level splits.")



def main():
    args = parse_args()
    set_seed(args.seed)
    model_key = str(args.model_name).strip().lower()
    if args.dataset_path is None:
        if model_key not in TRAIN_DATASET_RELATIVE_DEFAULTS:
            raise KeyError(
                f"No default training dataset for model {args.model_name!r}; pass --dataset-path explicitly."
            )
        args.dataset_path = TRAIN_DATASET_RELATIVE_DEFAULTS[model_key]
    args.dataset_path = resolve_local_path(args.dataset_path)
    if args.dataset_tag is None:
        args.dataset_tag = args.dataset_path.stem
    samples = load_samples(args.dataset_path)
    if args.train_attack:
        filtered = []
        for sample in samples:
            row = dict(sample)
            if str(row.get("method", "")).lower() != args.train_attack.lower():
                row.pop("J_prompt", None)
            if available_labels_for_sample(row):
                filtered.append(row)
        if not any(r.get("J_prompt") for r in filtered):
            raise ValueError("--train-attack matched no jailbreak prompts.")
        samples = filtered
        args.dataset_tag += "_" + sanitize_name(args.train_attack)
    if args.train_size > 0:
        samples = samples[: args.train_size]
    counts_before = prompt_counts(samples)
    samples = deduplicate_prompt_fields(samples)
    counts_after = prompt_counts(samples)
    print(f"[Info] Exact-prompt deduplication: {counts_before} -> {counts_after}")
    args.train_size = len(samples)
    artifact_dir = build_artifact_dir(args)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / "data_preparation.json", {"before": counts_before, "after": counts_after,
               "policy": "exact prompt per class; train priority; original JSON unchanged"})

    model_path = resolve_model_path(args.model_name, args.model_path)
    template_name = infer_template_name(args.model_name)
    print(f"[Info] Loading model from {model_path}")
    model, tokenizer, use_device_map_auto = load_model_and_tokenizer_for_online(
        str(model_path),
        args.device,
        args.auto_gpu_memory,
        args.auto_cpu_memory,
    )
    if use_device_map_auto:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    expected_forwards = sum(len(available_labels_for_sample(sample)) for sample in samples)
    print(
        "[Info] Extracting prefill hidden states "
        f"for {len(samples)} samples with {expected_forwards} available J/B/H forwards...",
        flush=True,
    )
    examples = prepare_training_examples(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        template_name=template_name,
        device=str(get_input_device(model)),
        num_tokens=args.prefill_num_tokens,
        progress_every=args.progress_every,
    )

    print("[Info] Training prefill MLP detector...")
    _model, meta, train_ids, val_ids, split_mode, validation_metrics = train_detector(examples, args, artifact_dir)
    if split_mode == "declared_split":
        fit_examples = [e for e in examples if str(e.get("split", "")).lower().strip() == "train"]
    else:
        fit_examples = [e for e in examples if int(e["sample_id"]) in set(train_ids)]
    sample_example = examples[0]
    num_layers = int(sample_example["layer_matrix_raw"].shape[0])
    hidden_dim = int(sample_example["layer_matrix_raw"].shape[1])

    centers = compute_centers_raw(fit_examples, args.prefill_num_tokens, num_layers, hidden_dim)
    scheme = fit_fisher(fit_examples, centers, args.prefill_num_tokens, num_layers,
                        rho=args.fisher_rho, variance_floor=args.fisher_variance_floor,
                        device=args.fisher_device)
    torch.save(scheme, artifact_dir / "prefill_scheme1_center_vectors_raw.pt")
    torch.save({c: torch.as_tensor(v, dtype=torch.float64) for c, v in centers.items()},
               artifact_dir / "prefill_token_layer_centers_raw.pt")
    td_topk = rank_fisher_layers(scheme, args.td_topk_m_select, min(args.td_topk_max_k, num_layers))
    write_json(artifact_dir / f"td_topk_layer_selection_m{args.td_topk_m_select}.json", td_topk)
    torch.save(td_topk, artifact_dir / f"td_topk_layer_selection_m{args.td_topk_m_select}.pt")

    write_json(
        artifact_dir / "offline_config.json",
        {
            "dataset_path": str(args.dataset_path),
            "train_attack": args.train_attack,
            "model_name": args.model_name,
            "model_path": str(model_path),
            "default_model_relative_path": str(MODEL_RELATIVE_DEFAULTS.get(str(args.model_name).lower(), "")),
            "template_name": template_name,
            "device": args.device,
            "use_device_map_auto": bool(use_device_map_auto),
            "auto_gpu_memory": args.auto_gpu_memory,
            "auto_cpu_memory": args.auto_cpu_memory,
            "split_mode": split_mode,
            "train_size": int(args.train_size),
            "num_train_rows": int(len(train_ids)),
            "num_val_rows": int(len(val_ids)),
            "prefill_num_tokens": int(args.prefill_num_tokens),
            "td_topk_m_select": int(args.td_topk_m_select),
            "td_topk_max_k": int(args.td_topk_max_k),
            "fisher_rho": args.fisher_rho,
            "fisher_variance_floor": args.fisher_variance_floor,
            "hidden_state_source": "raw_decoder_block_output",
            "seed": int(args.seed),
            "progress_every": int(args.progress_every),
        },
    )
    write_json(
        artifact_dir / "split_info.json",
        {
            "train_ids": train_ids,
            "val_ids": val_ids,
            "split_mode": split_mode,
            "label_order": list(LABEL_ORDER),
        },
    )
    write_json(
        artifact_dir / "offline_summary.json",
        {
            "artifact_dir": str(artifact_dir),
            "prefill_detector_meta": meta,
            "validation_metrics_path": str(artifact_dir / "prefill_validation_metrics.json"),
            "overall_val_acc": validation_metrics["overall_val_acc"],
            "val_acc_by_label": validation_metrics["val_acc_by_label"],
            "td_topk_path": str(artifact_dir / f"td_topk_layer_selection_m{args.td_topk_m_select}.json"),
        },
    )
    print(f"[Info] PRI training artifacts saved to {artifact_dir}")


if __name__ == "__main__":
    main()
