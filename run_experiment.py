from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from data_pipeline import (
    acquire_dataset,
    build_heuristic_labels,
    load_and_preprocess,
    make_pair_indices,
    write_provenance,
)
from evaluation import (
    binary_metrics,
    choose_threshold,
    paired_tests,
    rule_based_score,
    run_shap_on_prediction_head,
    summarize_metrics,
    tfidf_pair_score,
    topk_hit_rate,
    transformer_cosine_score,
)
from models import (
    DDQNPERAgent,
    PairGraphTransformer,
    TransformerTextEncoder,
    infer_matcher,
    resolve_device,
    set_global_seed,
    train_supervised_matcher,
)
from simulation import DynamicPairEnvironment, SimulatedSignals, generate_simulated_signals


def _tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)


def _make_model_inputs(
    bundle,
    text_embeddings,
    candidate_idx,
    project_idx,
    scalar_features,
    indices,
):
    idx = np.asarray(indices, dtype=np.int64)
    return (
        _tensor(bundle.skill_matrix[candidate_idx[idx]]),
        _tensor(bundle.skill_matrix[project_idx[idx]]),
        _tensor(text_embeddings[candidate_idx[idx]]),
        _tensor(text_embeddings[project_idx[idx]]),
        _tensor(scalar_features[idx]),
    )


def _split_development(
    dev_idx: np.ndarray,
    labels: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=seed,
    )
    train_rel, val_rel = next(splitter.split(dev_idx, labels[dev_idx]))
    return dev_idx[train_rel], dev_idx[val_rel]


def _write_split_rows(rows, seed, fold, train_idx, val_idx, test_idx):
    for split_name, indices in (
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx),
    ):
        rows.extend(
            {"seed": seed, "fold": fold, "row_index": int(i), "split": split_name}
            for i in indices
        )


def _calibrate_and_score(
    model_name,
    y_val,
    val_score,
    y_test,
    test_score,
    threshold_metric,
    seed,
    fold,
):
    threshold = choose_threshold(y_val, val_score, threshold_metric)
    metrics = binary_metrics(y_test, test_score, threshold)
    metrics.update(
        {"model": model_name, "seed": seed, "fold": fold, "threshold": threshold}
    )
    return metrics, threshold


def run(config: Dict, data_path: str | None, download: bool, output_dir: Path, smoke: bool = False):
    exp_cfg = config["experiment"]
    ds_cfg = config["dataset"]
    label_cfg = config["labels"]
    sim_cfg = config["simulation"]
    text_cfg = config["text_encoder"]
    model_cfg = config["model"]
    ddqn_cfg = config["ddqn"]
    eval_cfg = config["evaluation"]

    if smoke:
        exp_cfg = dict(exp_cfg)
        model_cfg = dict(model_cfg)
        ddqn_cfg = dict(ddqn_cfg)
        exp_cfg["seeds"] = [int(exp_cfg["seeds"][0])]
        exp_cfg["n_folds"] = min(2, int(exp_cfg["n_folds"]))
        model_cfg["supervised_epochs"] = min(2, int(model_cfg["supervised_epochs"]))
        ddqn_cfg["train_steps"] = min(300, int(ddqn_cfg["train_steps"]))
        ddqn_cfg["warmup_steps"] = min(64, int(ddqn_cfg["warmup_steps"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    csv_path, source_meta = acquire_dataset(
        data_path,
        download,
        ds_cfg["kaggle_slug"],
        ds_cfg.get("csv_glob", "*.csv"),
    )
    bundle = load_and_preprocess(csv_path, source_meta, ds_cfg)

    # One stable pair universe is created before CV; the splits are over pairs.
    pair_rng = np.random.default_rng(202603)
    candidate_idx, project_idx = make_pair_indices(len(bundle.frame), pair_rng)
    labels, compatibility, label_definition, label_components = build_heuristic_labels(
        bundle, candidate_idx, project_idx, label_cfg
    )
    write_provenance(bundle, output_dir, label_definition)

    class_counts = np.bincount(labels, minlength=2)
    if class_counts.min() < int(exp_cfg["n_folds"]):
        raise ValueError(
            f"Insufficient samples per class for stratified {exp_cfg['n_folds']}-fold CV: "
            f"class counts={class_counts.tolist()}. Adjust label threshold or inspect dataset."
        )

    device = resolve_device(exp_cfg.get("device", "auto"))

    # Text embeddings are deterministic fixed representations for all folds.
    text_encoder = TransformerTextEncoder(
        text_cfg["model_name"],
        max_length=text_cfg["max_length"],
        batch_size=text_cfg["batch_size"],
        device=str(device),
        allow_tfidf_fallback=text_cfg.get("allow_tfidf_fallback", False),
    )
    text_embeddings = text_encoder.encode(bundle.row_text)

    # Interpretable pair features. The first four are dataset-derived; the next
    # six are explicitly simulated.
    dataset_scalars = np.column_stack(
        [
            label_components["skill_similarity"],
            label_components["role_match"],
            label_components["course_match"],
            label_components["demand_alignment"],
        ]
    ).astype(np.float32)
    dataset_scalar_names = [
        "skill_overlap",
        "role_appropriateness",
        "course_alignment",
        "demand_alignment",
    ]

    fold_metric_rows = []
    topk_rows = []
    prediction_rows = []
    split_rows = []
    shap_done = False

    seeds = [int(s) for s in exp_cfg["seeds"]]
    n_folds = int(exp_cfg["n_folds"])
    val_frac = float(exp_cfg["validation_fraction_of_development"])

    for seed in seeds:
        set_global_seed(seed, bool(exp_cfg.get("deterministic", True)))
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        # Same simulated feature draw is used across folds of a seed so only
        # training/test membership changes, not the underlying seed scenario.
        sim_rng = np.random.default_rng(seed + 1000)
        sim_signals_obj = generate_simulated_signals(
            len(labels),
            sim_rng,
            sim_cfg,
            skill_similarity=label_components["skill_similarity"],
        )
        sim_matrix = sim_signals_obj.as_matrix()
        scalar_features = np.concatenate([dataset_scalars, sim_matrix], axis=1)
        scalar_names = dataset_scalar_names + SimulatedSignals.names()

        for fold, (dev_idx, test_idx) in enumerate(
            cv.split(np.arange(len(labels)), labels), start=1
        ):
            train_idx, val_idx = _split_development(
                dev_idx, labels, val_frac, seed * 100 + fold
            )
            _write_split_rows(split_rows, seed, fold, train_idx, val_idx, test_idx)

            train_inputs = _make_model_inputs(
                bundle, text_embeddings, candidate_idx, project_idx, scalar_features, train_idx
            )
            val_inputs = _make_model_inputs(
                bundle, text_embeddings, candidate_idx, project_idx, scalar_features, val_idx
            )
            test_inputs = _make_model_inputs(
                bundle, text_embeddings, candidate_idx, project_idx, scalar_features, test_idx
            )

            # ---------------- Baseline: Rule-based ----------------
            val_rule = rule_based_score(label_components, val_idx)
            test_rule = rule_based_score(label_components, test_idx)
            metrics, threshold = _calibrate_and_score(
                "Rule_based",
                labels[val_idx],
                val_rule,
                labels[test_idx],
                test_rule,
                eval_cfg["threshold_metric"],
                seed,
                fold,
            )
            fold_metric_rows.append(metrics)

            # ---------------- Baseline: TF-IDF ----------------
            val_tfidf = tfidf_pair_score(
                bundle.row_text, candidate_idx, project_idx, train_idx, val_idx
            )
            test_tfidf = tfidf_pair_score(
                bundle.row_text, candidate_idx, project_idx, train_idx, test_idx
            )
            metrics, _ = _calibrate_and_score(
                "TFIDF_Cosine",
                labels[val_idx],
                val_tfidf,
                labels[test_idx],
                test_tfidf,
                eval_cfg["threshold_metric"],
                seed,
                fold,
            )
            fold_metric_rows.append(metrics)

            # ---------------- Baseline: Transformer only ----------------
            val_bert = transformer_cosine_score(
                text_embeddings, candidate_idx, project_idx, val_idx
            )
            test_bert = transformer_cosine_score(
                text_embeddings, candidate_idx, project_idx, test_idx
            )
            metrics, _ = _calibrate_and_score(
                "Transformer_Cosine",
                labels[val_idx],
                val_bert,
                labels[test_idx],
                test_bert,
                eval_cfg["threshold_metric"],
                seed,
                fold,
            )
            fold_metric_rows.append(metrics)

            # ---------------- GNN-only ablation ----------------
            gnn_only = PairGraphTransformer(
                n_skills=len(bundle.skill_columns),
                text_dim=text_embeddings.shape[1],
                scalar_dim=scalar_features.shape[1],
                hidden_dim=int(model_cfg["hidden_dim"]),
                graph_heads=int(model_cfg["graph_heads"]),
                transformer_heads=int(model_cfg["transformer_heads"]),
                transformer_layers=int(model_cfg["transformer_layers"]),
                dropout=float(model_cfg["dropout"]),
                skill_threshold=float(model_cfg["graph_skill_threshold"]),
                use_transformer=False,
            )
            train_supervised_matcher(
                gnn_only,
                train_inputs,
                torch.as_tensor(labels[train_idx], dtype=torch.float32),
                val_inputs,
                torch.as_tensor(labels[val_idx], dtype=torch.float32),
                model_cfg,
                device,
            )
            val_gnn, _ = infer_matcher(gnn_only, val_inputs, device)
            test_gnn, _ = infer_matcher(gnn_only, test_inputs, device)
            metrics, _ = _calibrate_and_score(
                "GNN_only",
                labels[val_idx],
                val_gnn,
                labels[test_idx],
                test_gnn,
                eval_cfg["threshold_metric"],
                seed,
                fold,
            )
            fold_metric_rows.append(metrics)

            # ---------------- Full supervised GNN + Transformer ----------------
            full_model = PairGraphTransformer(
                n_skills=len(bundle.skill_columns),
                text_dim=text_embeddings.shape[1],
                scalar_dim=scalar_features.shape[1],
                hidden_dim=int(model_cfg["hidden_dim"]),
                graph_heads=int(model_cfg["graph_heads"]),
                transformer_heads=int(model_cfg["transformer_heads"]),
                transformer_layers=int(model_cfg["transformer_layers"]),
                dropout=float(model_cfg["dropout"]),
                skill_threshold=float(model_cfg["graph_skill_threshold"]),
                use_transformer=True,
            )
            train_result = train_supervised_matcher(
                full_model,
                train_inputs,
                torch.as_tensor(labels[train_idx], dtype=torch.float32),
                val_inputs,
                torch.as_tensor(labels[val_idx], dtype=torch.float32),
                model_cfg,
                device,
            )
            val_sup, val_latent = infer_matcher(full_model, val_inputs, device)
            test_sup, test_latent = infer_matcher(full_model, test_inputs, device)
            train_sup, train_latent = infer_matcher(full_model, train_inputs, device)

            # ---------------- DDQN-PER adaptive module ----------------
            train_states_base = np.concatenate(
                [train_latent, sim_matrix[train_idx]], axis=1
            ).astype(np.float32)
            val_states_base = np.concatenate(
                [val_latent, sim_matrix[val_idx]], axis=1
            ).astype(np.float32)
            test_states_base = np.concatenate(
                [test_latent, sim_matrix[test_idx]], axis=1
            ).astype(np.float32)

            env = DynamicPairEnvironment(
                train_latent,
                labels[train_idx],
                sim_matrix[train_idx],
                np.random.default_rng(seed * 1000 + fold),
                sim_cfg,
            )
            agent = DDQNPERAgent(
                state_dim=env.state_dim,
                cfg=ddqn_cfg,
                device=device,
                seed=seed * 1000 + fold,
            )
            agent.train_in_environment(env)

            val_adv = agent.score_assign_advantage(val_states_base)
            test_adv = agent.score_assign_advantage(test_states_base)

            rl_weight = float(ddqn_cfg["proposed_rl_weight"])
            val_proposed = (1.0 - rl_weight) * val_sup + rl_weight * (
                1.0 / (1.0 + np.exp(-val_adv))
            )
            test_proposed = (1.0 - rl_weight) * test_sup + rl_weight * (
                1.0 / (1.0 + np.exp(-test_adv))
            )

            metrics, proposed_threshold = _calibrate_and_score(
                "Proposed_GNN_Transformer_DDQN_PER",
                labels[val_idx],
                val_proposed,
                labels[test_idx],
                test_proposed,
                eval_cfg["threshold_metric"],
                seed,
                fold,
            )
            fold_metric_rows.append(metrics)

            # Top-k on the same test scores. Project source row acts as a stable
            # synthetic project identifier in the semi-synthetic protocol.
            for k in eval_cfg.get("topk", [1, 3, 5]):
                topk_rows.append(
                    {
                        "model": "Proposed_GNN_Transformer_DDQN_PER",
                        "seed": seed,
                        "fold": fold,
                        "k": int(k),
                        "hit_rate": topk_hit_rate(
                            labels[test_idx],
                            test_proposed,
                            project_idx[test_idx],
                            int(k),
                        ),
                    }
                )

            for local, row_index in enumerate(test_idx):
                prediction_rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "row_index": int(row_index),
                        "candidate_source_index": int(candidate_idx[row_index]),
                        "project_source_index": int(project_idx[row_index]),
                        "heuristic_label": int(labels[row_index]),
                        "heuristic_compatibility": float(compatibility[row_index]),
                        "supervised_score": float(test_sup[local]),
                        "ddqn_assign_advantage": float(test_adv[local]),
                        "proposed_score": float(test_proposed[local]),
                        "proposed_prediction": int(test_proposed[local] >= proposed_threshold),
                    }
                )

            checkpoint = {
                "seed": seed,
                "fold": fold,
                "model_state": full_model.state_dict(),
                "ddqn_online_state": agent.online.state_dict(),
                "skill_columns": bundle.skill_columns,
                "scalar_names": scalar_names,
                "text_model_name": text_cfg["model_name"],
            }
            torch.save(
                checkpoint,
                output_dir / f"checkpoint_seed{seed}_fold{fold}.pt",
            )

            # SHAP once on the first seed/fold, using validation background and
            # held-out test explanations. Latents are fixed before explanation.
            if not shap_done and not smoke:
                # Construct a full latent/scalar matrix only for the two required partitions.
                latent_join = np.vstack([val_latent, test_latent])
                scalar_join = np.vstack([scalar_features[val_idx], scalar_features[test_idx]])
                val_local = np.arange(len(val_idx))
                test_local = np.arange(len(val_idx), len(val_idx) + len(test_idx))
                run_shap_on_prediction_head(
                    full_model.prediction_head,
                    latent_join,
                    scalar_join,
                    scalar_names,
                    val_local,
                    test_local,
                    output_dir / "shap_feature_importance.csv",
                    background_size=int(eval_cfg.get("shap_background", 500)),
                    explain_size=int(eval_cfg.get("shap_explain", 500)),
                    device=device,
                )
                shap_done = True

    fold_metrics = pd.DataFrame(fold_metric_rows)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output_dir / "split_manifest.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(topk_rows).to_csv(output_dir / "topk_metrics.csv", index=False)

    summary = summarize_metrics(
        fold_metrics, confidence_level=float(eval_cfg["confidence_level"])
    )
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)

    tests = paired_tests(
        fold_metrics,
        proposed_name="Proposed_GNN_Transformer_DDQN_PER",
        metric="f1",
    )
    tests.to_csv(output_dir / "paired_tests.csv", index=False)

    return output_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--data", default=None)
    p.add_argument("--download", action="store_true")
    p.add_argument("--output", required=True)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run(
        config=config,
        data_path=args.data,
        download=args.download,
        output_dir=Path(args.output),
        smoke=args.smoke,
    )
