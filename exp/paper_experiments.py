"""Offline data audit, held-out detector accuracy, and paired inference overhead.

No API calls. Timing includes tokenization, detection and conditional second
prefill; CUDA is synchronized at measurement boundaries. Model loading and
warm-up are excluded. This script never synthesizes paper result numbers.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate_asr as runtime
from data_registry import load_attack_samples
from train_pri_detector import load_samples, deduplicate_prompt_fields, prompt_counts


def synchronize():
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def audit(model_name):
    from data_registry import DATA_DIR, model_file_for
    path = DATA_DIR / "train" / ("comprehensive_" + model_file_for(model_name)[:-5] + "_train.json")
    samples = load_samples(path)
    clean = deduplicate_prompt_fields(samples)
    return {"model_name": model_name, "dataset": str(path),
            "raw_counts": prompt_counts(samples), "deduplicated_counts": prompt_counts(clean),
            "heldout_counts": {a: len(load_attack_samples(a, model_name, split="test"))
                               for a in ("gcg", "saa", "autodan", "pair", "drattack",
                                         "template", "deepinception", "sap30", "advbench", "hex_phi")},
            "note": "Counts reflect files on disk, not target counts asserted in the manuscript."}


def load_context(args):
    runtime.set_seed(args.seed)
    model, tokenizer, auto = runtime.load_model_and_tokenizer_for_online(
        str(runtime.resolve_model_path(args.model_name, args.model_path)),
        args.device, args.auto_gpu_memory, args.auto_cpu_memory)
    detector_device = "cpu" if auto else str(runtime.get_input_device(model))
    artifact_dir = args.artifact_dir or runtime.latest_artifact_dir(args.training_root, args.model_name)
    artifacts = runtime.load_artifacts(artifact_dir, detector_device)
    m = int(artifacts["prefill_meta"]["num_tokens"])
    selection_args = argparse.Namespace(td_topk_layer_selection_path=None, prefill_num_tokens=m)
    path = runtime.resolve_td_topk_selection_path(artifact_dir, selection_args, artifacts)
    layers, _, _, _ = runtime.load_td_topk_selected_layers(
        path, args.td_topk_k, int(artifacts["prefill_meta"]["num_layers"]),
        fisher_rho=artifacts["prefill_scheme"]["rho"], num_tokens=m)
    return model, tokenizer, artifacts, m, layers, detector_device, str(artifact_dir)


def detect(model, tokenizer, artifacts, sample, template, m, device):
    inputs = runtime.build_inputs(tokenizer, template, sample["instruction"],
                                  str(runtime.get_input_device(model)),
                                  whitebox_attacker=runtime.is_prompt_optimized_attack(sample["attack"]),
                                  use_fastchat_template=runtime.uses_fastchat_template(sample["attack"]))
    with torch.inference_mode():
        outputs = runtime.forward_current(model, **runtime.filter_model_inputs_online(model, inputs),
                                          output_hidden_states=True)
        matrices = runtime.stack_prefill_hidden_states(outputs, m)
        return [runtime.classify_token(artifacts["prefill_detector"], x, device)["pred_label"] for x in matrices]


def measure(args, context, sample, defended):
    model, tokenizer, artifacts, m, layers, device, _ = context
    template = runtime.infer_template_name(args.model_name)
    synchronize()
    start = time.perf_counter()
    first = []
    def first_token():
        synchronize()
        first.append(time.perf_counter() - start)
    with torch.inference_mode():
        if defended:
            _, count, trace, _ = runtime.generate_with_pri_prefill_only(
                model, tokenizer, artifacts, sample, template, layers, m,
                args.max_new_tokens, False, None, device, first_token_callback=first_token)
            triggered = any(row["apply_mitigation"] for row in trace)
        else:
            inputs = runtime.build_inputs(tokenizer, template, sample["instruction"],
                                          str(runtime.get_input_device(model)),
                                          whitebox_attacker=runtime.is_prompt_optimized_attack(sample["attack"]),
                                          use_fastchat_template=runtime.uses_fastchat_template(sample["attack"]))
            inputs = runtime.filter_model_inputs_online(model, inputs)
            outputs = runtime.forward_current(model, **inputs)
            ids = runtime.decode_from_prefill(model, tokenizer, outputs, inputs["input_ids"],
                                              inputs.get("attention_mask"), inputs.get("token_type_ids"),
                                              args.max_new_tokens, False, None, first_token)
            count, triggered = len(ids), False
            tokenizer.decode(ids, skip_special_tokens=False)
    synchronize()
    total = time.perf_counter() - start
    return {"seconds": total, "tokens": count, "seconds_per_token": ratio(total, count),
            "ttft_seconds": first[0], "triggered": triggered}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("mode", choices=["audit", "detector", "overhead"])
    p.add_argument("--model-name", default="vicuna-7b")
    p.add_argument("--model-path", type=Path)
    p.add_argument("--artifact-dir", type=Path)
    p.add_argument("--training-root", type=Path, default=runtime.ROOT_DIR / "training_results")
    p.add_argument("--attack", default="gcg", help="Comma-separated datasets; overhead accepts a balanced 240-input JSON via --dataset-path.")
    p.add_argument("--dataset-path", type=Path)
    p.add_argument("--data-split", choices=["test", "all"], default="test",
                   help="Use all only for a supplied independently held-out custom file.")
    p.add_argument("--expected-label", choices=["J", "B", "H"], default=None)
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--td-topk-k", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=runtime.default_device())
    p.add_argument("--auto-gpu-memory", default="22GiB")
    p.add_argument("--auto-cpu-memory", default="64GiB")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--output", type=Path, default=runtime.ROOT_DIR / "results" / "paper_experiments.json")
    args = p.parse_args()
    if args.mode == "audit":
        report = audit(args.model_name)
    else:
        if args.max_new_tokens <= 0 or args.repeats <= 0 or args.warmup < 0:
            p.error("Require positive generation length/repeats and nonnegative warmup.")
        context = load_context(args)
        records = []
        for attack in args.attack.split(","):
            samples = load_attack_samples(attack, args.model_name, num_samples=args.num_samples,
                                          dataset_path=args.dataset_path, split=args.data_split, seed=args.seed)
            if not samples:
                raise ValueError(f"No held-out samples for {attack}.")
            if args.mode == "overhead":
                for _ in range(args.warmup):
                    measure(args, context, samples[0], False)
                    measure(args, context, samples[0], True)
            for sample in samples:
                record = {"sample_uid": sample["sample_uid"], "attack": attack}
                if args.mode == "detector":
                    labels = detect(context[0], context[1], context[2], sample,
                                    runtime.infer_template_name(args.model_name), context[3], context[5])
                    expected = args.expected_label or ("H" if attack in {"advbench", "hex_phi"} else "J")
                    record.update(expected_label=expected, labels=labels, triggered="J" in labels,
                                  correct_tokens=labels.count(expected), num_tokens=len(labels))
                else:
                    pairs = []
                    for repeat in range(args.repeats):
                        pair = {}
                        # Alternate order to reduce systematic warm-cache/order bias.
                        for defended in ([False, True] if repeat % 2 == 0 else [True, False]):
                            pair["fishersteer" if defended else "baseline"] = measure(args, context, sample, defended)
                        pairs.append(pair)
                    record["measurements"] = pairs
                records.append(record)
                print(f"[{args.mode}] {sample['sample_uid']}", flush=True)
        report = {"mode": args.mode, "model_name": args.model_name, "artifact_dir": context[-1],
                  "seed": args.seed, "selected_layers": context[4], "data": records}
        if args.mode == "detector":
            report["token_accuracy"] = ratio(sum(r["correct_tokens"] for r in records), sum(r["num_tokens"] for r in records))
            benign = [r for r in records if r["expected_label"] == "B"]
            report["prompt_FPR"] = ratio(sum(r["triggered"] for r in benign), len(benign))
            report["predicted_token_counts"] = dict(Counter(c for r in records for c in r["labels"]))
            report["token_accuracy_by_attack"] = {
                a: ratio(sum(r["correct_tokens"] for r in records if r["attack"] == a),
                         sum(r["num_tokens"] for r in records if r["attack"] == a))
                for a in sorted({r["attack"] for r in records})}
        else:
            pairs = [pair for r in records for pair in r["measurements"]]
            for metric, name in [("ttft_seconds", "TTFT_ratio"), ("seconds_per_token", "ATGR")]:
                means = {method: float(np.mean([pair[method][metric] for pair in pairs]))
                         for method in ("baseline", "fishersteer")}
                report[name] = ratio(means["fishersteer"], means["baseline"])
                report["mean_" + metric] = means
    runtime.write_json(args.output, report)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
