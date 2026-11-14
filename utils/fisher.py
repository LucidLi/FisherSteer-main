"""Exact, class-balanced regularized Fisher calibration (paper Sections 3.2-3.3)."""
import numpy as np
import torch

LABELS = ("J", "B", "H")
FISHER_FORMAT = 2


def fit_fisher(examples, centers, num_tokens, num_layers, rho=0.1, variance_floor=1e-12, device="cpu"):
    if not np.isfinite(rho) or rho <= 0 or not np.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("Fisher rho and variance floor must be positive and finite.")
    shape = centers["H"].shape
    directions = {c: np.zeros(shape, dtype=np.float64) for c in ("J", "B")}
    scores = {c: np.zeros(shape[:2], dtype=np.float64) for c in ("J", "B")}
    tau_values = np.zeros(shape[:2], dtype=np.float64)
    grouped = {(c, t): [] for c in LABELS for t in range(num_tokens)}
    for ex in examples:
        grouped[ex["label"], int(ex["token_index"]) - 1].append(ex["layer_matrix_raw"])
    for t in range(num_tokens):
        if any(not grouped[c, t] for c in LABELS):
            raise ValueError(f"All three training classes are required at relative token {t + 1}.")
        for layer in range(num_layers):
            # A.T @ A = (Sigma_J + Sigma_B + Sigma_H) / 3, with 1/N_c covariance.
            blocks = []
            for c in LABELS:
                x = np.stack([x[layer] for x in grouped[c, t]]).astype(np.float64)
                blocks.append((x - centers[c][t, layer]) / np.sqrt(3.0 * len(x)))
            a = torch.as_tensor(np.concatenate(blocks), dtype=torch.float64, device=device)
            tau = float(a.square().sum().item() / shape[-1])
            tau_values[t, layer] = tau
            # tau=0 violates the paper's SPD assumption; use an explicit numerical floor.
            ridge = rho * max(tau, variance_floor)
            v = torch.as_tensor(np.stack([centers["H"][t, layer] - centers[c][t, layer]
                                          for c in ("J", "B")], axis=1), dtype=torch.float64, device=device)
            if a.shape[0] < a.shape[1]:
                # Woodbury identity, exact full covariance; never a diagonal approximation.
                gram = a @ a.T
                gram.diagonal().add_(ridge)
                w = (v - a.T @ torch.linalg.solve(gram, a @ v)) / ridge
            else:
                cov = a.T @ a
                cov.diagonal().add_(ridge)
                w = torch.linalg.solve(cov, v)
            if not torch.isfinite(w).all():
                raise ValueError("Non-finite Fisher solve; inspect calibration inputs/rho.")
            for j, c in enumerate(("J", "B")):
                directions[c][t, layer] = w[:, j].cpu().numpy()
                scores[c][t, layer] = max(0.0, float(torch.dot(v[:, j], w[:, j]).item()))
        print(f"[Fisher] calibrated relative token {t + 1}/{num_tokens}", flush=True)
    scheme = {"format_version": torch.tensor(FISHER_FORMAT),
              "rho": torch.tensor(rho, dtype=torch.float64),
              "variance_floor": torch.tensor(variance_floor, dtype=torch.float64),
              "tau": torch.from_numpy(tau_values)}
    for c in LABELS:
        scheme[f"mu_{c}"] = torch.as_tensor(centers[c], dtype=torch.float64)
    for c in ("J", "B"):
        scheme[f"v_{c}_to_H_raw"] = scheme["mu_H"] - scheme[f"mu_{c}"]
        scheme[f"w_{c}_to_H"] = torch.from_numpy(directions[c])
        scheme[f"R_H{c}"] = torch.from_numpy(scores[c])
    return scheme


def rank_fisher_layers(scheme, m_select, max_k):
    r = scheme["R_HJ"] + scheme["R_HB"]
    if not 1 <= m_select <= r.shape[0] or not 1 <= max_k <= r.shape[1]:
        raise ValueError("Fisher m_select/max_k are outside the calibrated dimensions.")
    s = r[:m_select].mean(dim=0).tolist()
    ranked = sorted([{"layer": i + 1, "score": float(v)} for i, v in enumerate(s)],
                    key=lambda item: (-item["score"], item["layer"]))
    return {"algorithm": "FisherSteer regularized Fisher", "format_version": FISHER_FORMAT,
            "m_select": m_select, "max_k": max_k, "num_layers": len(s),
            "rho": float(scheme["rho"]),
            "score_definition": "mean_t(R_HJ[t,l] + R_HB[t,l])",
            "topk_layers": {f"k={k}": sorted(x["layer"] for x in ranked[:k]) for k in range(1, max_k + 1)},
            "topk_layers_ranked_by_score": {f"k={k}": [x["layer"] for x in ranked[:k]] for k in range(1, max_k + 1)},
            "ranked_layers": ranked, "scores_by_layer": {str(x["layer"]): x for x in ranked}}
