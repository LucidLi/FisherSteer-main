"""Numerical and CPU integration checks; no model download or API call."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "exp")]
from utils.fisher import fit_fisher, rank_fisher_layers


class FisherMathTests(unittest.TestCase):
    def check_solve(self, dim):
        rng = np.random.default_rng(42)
        examples, centers, matrices = [], {}, {}
        for c, n in zip(("J", "B", "H"), (3, 4, 5)):
            x = rng.normal(size=(n, 2, dim)) + (2 if c == "H" else 0)
            matrices[c] = x
            centers[c] = x.mean(0)[None]
            examples.extend({"label": c, "token_index": 1, "layer_matrix_raw": v} for v in x)
        scheme = fit_fisher(examples, centers, 1, 2, rho=.2)
        for layer in range(2):
            cov = sum((matrices[c][:, layer] - centers[c][0, layer]).T @
                      (matrices[c][:, layer] - centers[c][0, layer]) / len(matrices[c])
                      for c in matrices) / 3
            cov += .2 * np.trace(cov) / dim * np.eye(dim)
            for c in ("J", "B"):
                v = centers["H"][0, layer] - centers[c][0, layer]
                expected = np.linalg.solve(cov, v)
                np.testing.assert_allclose(scheme[f"w_{c}_to_H"][0, layer], expected, rtol=1e-9, atol=1e-9)
                self.assertAlmostEqual(float(scheme[f"R_H{c}"][0, layer]), v @ expected, places=8)
        ranked = rank_fisher_layers(scheme, 1, 2)
        self.assertEqual(ranked["topk_layers"]["k=2"], [1, 2])
    def test_primal_full_balanced_covariance(self):
        self.check_solve(3)
    def test_woodbury_full_balanced_covariance(self):
        self.check_solve(20)
    def test_zero_variance_and_identical_means(self):
        centers = {c: np.ones((1, 1, 3)) for c in ("J", "B", "H")}
        ex = [{"label": c, "token_index": 1, "layer_matrix_raw": np.ones((1, 3))} for c in centers]
        result = fit_fisher(ex, centers, 1, 1)
        self.assertEqual(float(result["R_HJ"][0, 0]), 0)
        self.assertTrue(torch.isfinite(result["w_J_to_H"]).all())
    def test_missing_class_rejected(self):
        centers = {c: np.zeros((1, 1, 2)) for c in ("J", "B", "H")}
        with self.assertRaises(ValueError):
            fit_fisher([], centers, 1, 1)

    def test_default_rho_is_point_one_and_changes_the_solve(self):
        centers = {"J": np.array([[[0., 0.]]]), "B": np.array([[[1., 0.]]]),
                   "H": np.array([[[2., 1.]]])}
        examples = []
        cov = np.array([[1., .5], [.5, .5]])
        # Two orthogonal deviations give the covariance above under 1/N.
        deviations = np.array([[1., 1.], [-1., -1.], [1., 0.], [-1., 0.]])
        for c in centers:
            examples.extend({"label": c, "token_index": 1,
                             "layer_matrix_raw": centers[c][0] + x[None]} for x in deviations)
        scheme = fit_fisher(examples, centers, 1, 1)
        self.assertEqual(float(scheme["rho"]), .1)
        regularized = cov + .1 * np.trace(cov) / 2 * np.eye(2)
        expected = np.linalg.solve(regularized, np.array([2., 1.]))
        np.testing.assert_allclose(scheme["w_J_to_H"][0, 0], expected, rtol=1e-12)
        other = fit_fisher(examples, centers, 1, 1, rho=1.)
        self.assertFalse(torch.allclose(scheme["w_J_to_H"], other["w_J_to_H"]))

    def test_layer_scores_average_both_pairs_over_positions(self):
        scheme = {"R_HJ": torch.tensor([[10., 0., 2.], [0., 6., 2.]]),
                  "R_HB": torch.tensor([[0., 0., 1.], [0., 6., 1.]]),
                  "rho": torch.tensor(.1, dtype=torch.float64)}
        record = rank_fisher_layers(scheme, 2, 2)
        self.assertEqual([r["score"] for r in record["ranked_layers"]], [6., 5., 3.])
        self.assertEqual(record["topk_layers_ranked_by_score"]["k=2"], [2, 1])


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import evaluate_asr as runtime
        import train_pri_detector as train
        cls.runtime, cls.train = runtime, train

    def test_fisher_projection_constraint(self):
        v = torch.tensor([2., 1.], dtype=torch.float64)
        w = torch.tensor([.2, 3.], dtype=torch.float64)
        target, h = torch.tensor([1., 2.]), torch.tensor([-2., 5.])
        lam = self.runtime.compute_projection_lambda(h, target, v, w)
        self.assertAlmostEqual(float(w @ (h + lam * v - target)), 0, places=12)
        self.assertEqual(self.runtime.compute_projection_lambda(h, target, v * 0, w * 0), 0)

    def test_current_state_multilayer_hooks_and_cleanup(self):
        class Block(torch.nn.Module):
            def forward(self, x):
                return x * 2
        model = SimpleNamespace(model=SimpleNamespace(layers=torch.nn.ModuleList([Block(), Block()])))
        shape = (1, 2, 2)
        scheme = {"v_J_to_H_raw": torch.ones(shape, dtype=torch.float64),
                  "w_J_to_H": torch.ones(shape, dtype=torch.float64),
                  "mu_H": torch.full(shape, 3., dtype=torch.float64)}
        info = {0: {"apply_mitigation": True, "pred_label": "J", "token_index": 1}}
        records = []
        with self.runtime.apply_fisher_prefill_hooks(model, info, scheme, [1, 2], records):
            x = torch.zeros((1, 2, 2))
            for layer in model.model.layers:
                x = layer(x)
        torch.testing.assert_close(x[0, 0], torch.tensor([3., 3.]))
        torch.testing.assert_close(x[0, 1], torch.zeros(2))
        self.assertEqual([r["projected_lambda"] for r in records[0]["layers"]], [3., -3.])
        self.assertTrue(all(not layer._forward_hooks for layer in model.model.layers))

    def test_real_training_files_no_prompt_leakage(self):
        for path in (ROOT / "data/train").glob("*.json"):
            samples = self.train.deduplicate_prompt_fields(self.train.load_samples(path))
            train, val = [], []
            for row in samples:
                for c in self.train.available_labels_for_sample(row):
                    ex = {"sample_id": row["id"], "prompt_key": row[c + "_prompt"]}
                    (train if row.get("split") == "train" else val).append(ex)
            self.train.validate_prompt_disjoint(train, val)

    def test_legacy_selection_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / "old.json"
            path.write_text(json.dumps({"topk_layers": {"k=1": [1]}}))
            with self.assertRaisesRegex(ValueError, "Legacy"):
                self.runtime.load_td_topk_selected_layers(path, 1, 2)

    def test_detector_training_and_new_artifact_load(self):
        rng = np.random.default_rng(42)
        examples = []
        for i in range(12):
            c = ("J", "B", "H")[i % 3]
            for t in (1, 2):
                examples.append({"sample_id": i, "split": "train" if i < 9 else "val",
                                 "prompt_key": str(i), "token_index": t, "label": c,
                                 "y": i % 3, "layer_matrix_raw": rng.normal(size=(2, 4)).astype(np.float16)})
        args = SimpleNamespace(val_ratio=.2, seed=42, mlp_hidden_dim=8, mlp_encoder_out_dim=4,
                               mlp_dropout=.2, device="cpu", mlp_lr=.001, mlp_weight_decay=.0001,
                               mlp_epochs=1, mlp_batch_size=4, prefill_num_tokens=2)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp)
            self.train.train_detector(examples, args, path)
            fit = [e for e in examples if e["split"] == "train"]
            centers = self.train.compute_centers_raw(fit, 2, 2, 4)
            scheme = fit_fisher(fit, centers, 2, 2)
            torch.save(scheme, path / "prefill_scheme1_center_vectors_raw.pt")
            torch.save({c: torch.from_numpy(x) for c, x in centers.items()}, path / "prefill_token_layer_centers_raw.pt")
            artifacts = self.runtime.load_artifacts(path)
            result = self.runtime.classify_token(artifacts["prefill_detector"], np.ones((2, 4)))
            self.assertIn(result["pred_label"], ("J", "B", "H"))

    def test_cli_defaults_and_artifact_parameter_consistency(self):
        with patch.object(sys, "argv", ["train"]):
            args = self.train.parse_args()
        self.assertEqual(args.fisher_rho, .1)
        self.assertEqual(args.td_topk_m_select, args.prefill_num_tokens)
        scheme = {"R_HJ": torch.ones((2, 2)), "R_HB": torch.ones((2, 2)),
                  "rho": torch.tensor(.1, dtype=torch.float64)}
        record = rank_fisher_layers(scheme, 2, 1)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / "selection.json"
            path.write_text(json.dumps(record))
            self.runtime.load_td_topk_selected_layers(path, 1, 2, fisher_rho=.1, num_tokens=2)
            with self.assertRaisesRegex(ValueError, "rho"):
                self.runtime.load_td_topk_selected_layers(path, 1, 2, fisher_rho=.2)
            with self.assertRaisesRegex(ValueError, "same m"):
                self.runtime.load_td_topk_selected_layers(path, 1, 2, num_tokens=1)

    def test_training_and_inference_raw_features_match_without_extra_quantization(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)
        model = LlamaForCausalLM(config).eval()
        ids = torch.tensor([[1, 2, 3, 4]])
        with patch.object(self.train, "build_inputs", return_value={"input_ids": ids}):
            training = self.train.extract_one_prompt(model, None, "vicuna", "test", "cpu", 2)
        with torch.inference_mode():
            out = self.runtime.forward_current(model, ids, output_hidden_states=True)
        inference = self.runtime.stack_prefill_hidden_states(out, 2)
        for a, b in zip(training, inference):
            self.assertEqual(a.dtype, torch.float32)
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_mixed_labels_modify_only_j_and_defended_cache_matches_reference(self):
        from transformers import MistralConfig, MistralForCausalLM
        config = MistralConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
                               num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                               eos_token_id=None, pad_token_id=0)
        torch.manual_seed(123)
        model = MistralForCausalLM(config).eval()
        class Tokenizer:
            eos_token_id = None
            def decode(self, ids, **kwargs):
                return " ".join(str(i) for i in ids)
        shape = (3, 2, 16)
        scheme = {"mu_J": torch.zeros(shape), "mu_B": torch.zeros(shape),
                  "mu_H": torch.full(shape, .2), "v_J_to_H_raw": torch.ones(shape),
                  "w_J_to_H": torch.ones(shape)}
        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]]),
                  "attention_mask": torch.ones((1, 4), dtype=torch.long)}
        with patch.object(self.runtime, "build_inputs", return_value=inputs), patch.object(
                self.runtime, "classify_token", side_effect=[{"pred_label": c} for c in ("B", "J", "H")]) as detector:
            output, length, trace, records = self.runtime.generate_with_pri_prefill_only(
                model, Tokenizer(), {"prefill_scheme": scheme, "prefill_detector": {}},
                {"prompt": "test"}, "mistral", [1, 2], 3, 3, False, None, "cpu")
        self.assertEqual(detector.call_count, 3)  # no second detection or decoding-time monitor
        self.assertEqual([r["apply_mitigation"] for r in trace], [False, True, False])
        self.assertEqual([r["absolute_position"] for r in records], [2])
        for row in records[0]["layers"]:
            self.assertAlmostEqual(row["projection_gap_after"], 0, places=5)
        # Full-prefix reference recomputes the same prompt edits each step.
        # Matching it checks propagation of the edited prompt through the KV cache.
        reference = []
        ids = inputs["input_ids"]
        info = {2: {"token_index": 2, "pred_label": "J", "apply_mitigation": True}}
        with torch.inference_mode():
            for _ in range(3):
                with self.runtime.apply_fisher_prefill_hooks(model, info, scheme, [1, 2], []):
                    out = self.runtime.forward_current(model, ids)
                token = out.logits[:, -1].argmax(-1, keepdim=True)
                reference.append(int(token.item()))
                ids = torch.cat([ids, token], dim=1)
        self.assertEqual(output, Tokenizer().decode(reference))
        self.assertEqual(length, 3)
        self.assertTrue(all(not layer._forward_hooks for layer in model.model.layers))

    def test_tiny_llama_prefill_and_cache(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                             eos_token_id=None, pad_token_id=0)
        torch.manual_seed(42)
        model = LlamaForCausalLM(config).eval()
        calls = []
        handle = model.register_forward_pre_hook(lambda m, args, kwargs: calls.append(kwargs["input_ids"].shape[1]), with_kwargs=True)
        class Tokenizer:
            eos_token_id = None
            pad_token_id = 0
            def decode(self, ids, **kwargs):
                return " ".join(str(i) for i in ids)
        shape = (2, 2, 16)
        scheme = {"mu_J": torch.zeros(shape), "mu_B": torch.zeros(shape),
                  "mu_H": torch.ones(shape), "v_J_to_H_raw": torch.ones(shape),
                  "w_J_to_H": torch.ones(shape)}
        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]]), "attention_mask": torch.ones((1, 4), dtype=torch.long)}
        artifacts = {"prefill_scheme": scheme, "prefill_detector": {}}
        for label, expected_calls in (("B", [4, 1, 1]), ("J", [4, 4, 1, 1])):
            calls.clear()
            with patch.object(self.runtime, "build_inputs", return_value=inputs), patch.object(
                    self.runtime, "classify_token", return_value={"pred_label": label}):
                output, length, trace, records = self.runtime.generate_with_pri_prefill_only(
                    model, Tokenizer(), artifacts, {"prompt": "test"}, "vicuna", [1, 2],
                    2, 3, False, None, "cpu")
            self.assertEqual(length, 3)
            self.assertEqual(calls, expected_calls)
            self.assertEqual(bool(records), label == "J")
            self.assertTrue(all(not layer._forward_hooks for layer in model.model.layers))
        # Clean-cache continuation must match ordinary greedy generation exactly.
        handle.remove()
        clean = self.runtime.forward_current(model, **inputs, output_hidden_states=True)
        generated = self.runtime.decode_from_prefill(model, Tokenizer(), clean, inputs["input_ids"], inputs["attention_mask"], None, 3, False, None)
        reference = model.generate(**inputs, max_new_tokens=3, do_sample=False, use_cache=True)[0, 4:].tolist()
        self.assertEqual(generated, reference)


if __name__ == "__main__":
    unittest.main()
