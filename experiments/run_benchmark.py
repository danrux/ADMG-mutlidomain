"""Shared CLI construction and benchmark orchestration."""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    # Support direct execution as well as ``python -m experiments.run_benchmark``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import utils as ut
from data.registry import DGP_REGISTRY, generate_dataset
from evaluation import evaluate_graph
from evaluation.aggregation import flatten_metrics, summarize_metrics
from evaluation.truth import admg_reference_from_dataset
from experiments.result_store import RunResultStore
from graphs import ADMG, DirectedGraph
from methods.registry import BASELINE_REGISTRY, METHOD_REGISTRY
from evaluate import (
    compute_mcc,
    find_P_and_prune_fast,
    estimate_bidirected_from_Z_hat_list,
    select_bidirected_observations,
    align_z_hat,
)

def _prune_threshold(value):
    """Accept either the automatic pruning mode or a numeric threshold."""
    if value == "auto":
        return value
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be 'auto' or a floating-point number"
        ) from exc




def build_parser():
    """Build the CLI, grouped by the component that consumes each option."""
    parser = argparse.ArgumentParser(
        description="Multi-domain latent variable estimation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    selection = parser.add_argument_group("Method and data selection")
    selection.add_argument(
        "--method", "--baseline", dest="method", default="MuDo-nll",
        choices=list(BASELINE_REGISTRY),
        help="Estimation method/baseline.",
    )
    selection.add_argument(
        "--dgp", default="scaling", choices=list(DGP_REGISTRY),
        help="Data-generating process.",
    )

    experiment = parser.add_argument_group("Experiment")
    experiment.add_argument("--seeds", nargs="+", type=int, default=[2])
    experiment.add_argument("--evaluate", action="store_true", help="Evaluate saved results instead of fitting.")
    experiment.add_argument("--model-dir", default="outputs")
    data = parser.add_argument_group("Data generation (all DGPs)")
    data.add_argument("--T", type=int, default=1, help="Domain multiplier; ignored by the auxillary DGP.")# hard-intervention == n
    data.add_argument("--n", type=int, default=10, help="Number of latent variables.")
    data.add_argument("--x-n", type=int, default=1, help="Observed-dimension multiplier (observed dimension = x-n * n).")
    data.add_argument("--N", type=int, default=5000, help="Samples per domain.") #umni need >100000 per domain
    data.add_argument("--graph-dense", type=float, default=2.0)
    data.add_argument(
        "--confounder-dense",
        type=float,
        default=0.5,
        help=(
            "Latent-DAG edge multiplier (number of latent edges is approximately "
            "n * confounder-dense); ignored by DGPs without a latent DAG."
        ),
    )
    data.add_argument("--graph-type", default="ER")
    data.add_argument("--noise-type", default="gauss")
    data.add_argument(
        "--bow-free-ground-truth",
        action="store_true",
        help=(
            "For scaling, mask, and hard_intervention, constrain the observed "
            "DAG to node pairs without latent-induced bidirected edges."
        ),
    )

    mask_dgp = parser.add_argument_group("Mask DGPs (--dgp mask or nonlinear_mask)")
    mask_dgp.add_argument(
        "--mask-dense", type=float, default=0.5,
        help="Fraction of active latent dimensions per domain.",
    )

    intervention_dgp = parser.add_argument_group("Intervention DGP (--dgp hard_intervention)")
    intervention_dgp.add_argument(
        "--interv-dense", type=float, default=None,
        help="Fraction of intervened nodes per domain; defaults to 1/n for hard_intervention and 0.5 otherwise.",
    )

    optimization = parser.add_argument_group(
        "Shared neural optimization (MuDo-nll, SP, VaDE-linear, diagGMM)"
    )
    optimization.add_argument(
        "--lr", type=float, default=1e-2,
        help="Learning rate (typical: SP 1e-4, MuDo-nll 1e-2, gmm use 1e-3, UMNI does not use it).",
    )
    optimization.add_argument("--num-steps", type=int, default=5000)
    optimization.add_argument("--num-initializations", type=int, default=1)
    optimization.add_argument(
        "--batch-size", type=int, default=6144,
        help="Mini-batch size (SP, VaDE-linear, and diagGMM).",
    )
    optimization.add_argument("--log-every", type=int, default=250, help="Training-log interval.")

    mudo = parser.add_argument_group("MuDo-nll only (--method MuDo-nll)")
    mudo.add_argument("--loss-type", default="nll", choices=["cov", "nll"])
    mudo.add_argument("--scheduler", default="cosine", choices=["cosine", "plateau", "none"])
    mudo.add_argument("--grad-clip", type=float, default=1.0)
    mudo.add_argument("--patience", type=int, default=2000)

    sp = parser.add_argument_group("SP only (--method SP)")
    sp.add_argument("--sparse-level", type=float, default=0.01, help="Sparsity constraint threshold.")
    sp.add_argument(
        "--aug-lag-coef", type=float, default=0.0,
        help="Augmented Lagrangian coefficient for Cooper.",
    )

    umni = parser.add_argument_group("UMNI only (--method UMNI)")
    umni.add_argument("--atol-eigv", type=float, default=5e-2, help="Eigenvalue tolerance for rank detection.")
    umni.add_argument("--atol-ci-test", type=float, default=0.05, help="Partial-correlation CI-test p-value threshold.")
    umni.add_argument("--kappa", type=int, default=1, help="Maximum weight magnitude in the weight-set search.")
    umni.add_argument("--umni-batch-size", type=int, default=2048, help="Weight candidates per vectorized NumPy batch.")
    umni.add_argument("--umni-workers", type=int, default=0, help="Parallel CPU workers; 0 uses up to four cores.")

    vade = parser.add_argument_group("VaDE-linear only (--method VaDE-linear)")
    vade.add_argument("--vade-components", type=int, default=None, help="Mixture components; defaults to T.")
    vade.add_argument("--vade-recon-var", type=float, default=1.0, help="Gaussian decoder variance.")
    vade.add_argument("--vade-kl-weight", type=float, default=1.0, help="KL-term weight.")
    vade.add_argument("--vade-pretrain-steps", type=int, default=0, help="Autoencoder pretraining steps.")
    vade.add_argument("--vade-pretrain-lr", type=float, default=1e-3, help="Autoencoder pretraining learning rate.")
    vade.add_argument("--vade-gmm-init", action="store_true", help="Initialize the mixture prior with k-means.")
    vade.add_argument("--vade-kl-warmup-steps", type=int, default=0, help="Linear KL warmup duration.")
    vade.add_argument("--vade-full-cov-posterior", action="store_true", help="Use a full-covariance posterior.")

    exact_gmm = parser.add_argument_group("diagGMM only (--method diagGMM)")
    exact_gmm.add_argument("--exact-components", type=int, default=None, help="Mixture components; defaults to T.")
    exact_gmm.add_argument("--exact-jitter", type=float, default=1e-6, help="Numerical covariance jitter.")

    lingam = parser.add_argument_group("LiNGAM only (--baseline LiNGAM)")
    lingam.add_argument(
        "--lingam-measure",
        choices=["pwling", "kernel"],
        default="pwling",
        help="Independence measure used by DirectLiNGAM.",
    )
    lingam.add_argument(
        "--lingam-directed-threshold",
        type=float,
        default=0.0,
        help="Absolute threshold applied to DirectLiNGAM edge weights.",
    )
    lingam.add_argument(
        "--lingam-base-domain",
        choices=["auto", "last"],
        default="auto",
        help="Domain used to recover residuals for bidirected-edge estimation.",
    )
    lingam.add_argument(
        "--lingam-random-state",
        type=int,
        default=None,
        help="Optional internal DirectLiNGAM random seed.",
    )

    bang = parser.add_argument_group("BANG only (--baseline BANG)")
    bang.add_argument(
        "--bang-moment-degree", type=int, default=3,
        help="Maximum moment degree K; must be at least 3.",
    )
    bang.add_argument(
        "--bang-level", type=float, default=0.01,
        help="Empirical-likelihood test significance level.",
    )
    bang.add_argument(
        "--bang-restriction", type=int, choices=[1, 2], default=1,
        help="Moment family: 1 uses all degrees 2..K-1; 2 uses degree K-1 only.",
    )
    bang.add_argument(
        "--bang-domain-mode", choices=["pooled", "reference"], default="pooled",
        help="Stack all domains or use only the final (base) domain.",
    )
    bang.add_argument(
        "--bang-max-set-size", type=int, default=None,
        help="Optional cap on candidate parent-set size; the exact procedure is uncapped.",
    )
    bang.add_argument("--bang-max-iterations", type=int, default=10000)
    bang.add_argument(
        "--bang-condition-limit", type=float, default=1e12,
        help="Condition-number limit for debiasing systems.",
    )
    bang.add_argument(
        "--bang-directed-threshold", type=float, default=0.0,
        help="Optional absolute threshold applied after BANG estimates directed weights.",
    )
    bang.add_argument("--bang-verbose", action="store_true")

    dcd = parser.add_argument_group("DCD only (--baseline DCD)")
    dcd.add_argument(
        "--dcd-admg-class",
        choices=["bowfree", "ancestral", "arid"],
        default="bowfree",
        help="ADMG hypothesis class imposed by the differentiable constraint.",
    )
    dcd.add_argument(
        "--dcd-domain-mode", choices=["pooled", "reference"], default="pooled",
        help="Stack all domains or use only the final (base) domain.",
    )
    dcd.add_argument(
        "--dcd-standardize", action="store_true",
        help="Standardize each selected input column before fitting DCD.",
    )
    dcd.add_argument("--dcd-lambda", type=float, default=0.05, help="Approximate L0 penalty.")
    dcd.add_argument("--dcd-threshold", type=float, default=0.05, help="Edge threshold.")
    dcd.add_argument("--dcd-num-restarts", type=int, default=5)
    dcd.add_argument("--dcd-max-iterations", type=int, default=100)
    dcd.add_argument("--dcd-h-tol", type=float, default=1e-8, help="Graph-constraint tolerance.")
    dcd.add_argument("--dcd-rho-max", type=float, default=1e16)
    dcd.add_argument("--dcd-ricf-increment", type=int, default=1)
    dcd.add_argument("--dcd-ricf-tol", type=float, default=1e-4)
    dcd.add_argument("--dcd-ricf-refit-iterations", type=int, default=100)
    dcd.add_argument("--dcd-optimizer-max-iterations", type=int, default=15000)
    dcd.add_argument("--dcd-coefficient-bound", type=float, default=4.0)
    dcd.add_argument("--dcd-jitter", type=float, default=1e-8)
    dcd.add_argument(
        "--dcd-max-samples", type=int, default=None,
        help="Optional random cap on rows supplied to DCD.",
    )
    dcd.add_argument("--dcd-random-state", type=int, default=None)
    dcd.add_argument("--dcd-verbose", action="store_true")
    dcd.add_argument(
        "--skip-dcd-pag",
        action="store_true",
        help=(
            "Skip the optional causal-learn MAG/PAG conversion while retaining "
            "directed and bidirected F1, SHD, endpoint, weight, and MCC metrics."
        ),
    )

    rcd = parser.add_argument_group("RCD only (--baseline RCD)")
    rcd.add_argument(
        "--rcd-domain-mode", choices=["pooled", "reference"], default="pooled",
        help="Stack all domains or use only the final (base) domain.",
    )
    rcd.add_argument(
        "--rcd-max-samples", type=int, default=300,
        help="Random input-row cap; HSIC memory grows quadratically in this value.",
    )
    rcd.add_argument(
        "--rcd-max-explanatory-num", type=int, default=2,
        help="Maximum explanatory variables in RCD multiple regressions.",
    )
    rcd.add_argument("--rcd-cor-alpha", type=float, default=0.01)
    rcd.add_argument("--rcd-ind-alpha", type=float, default=0.01)
    rcd.add_argument("--rcd-shapiro-alpha", type=float, default=0.01)
    rcd.add_argument(
        "--rcd-mlhsicr", action="store_true",
        help="Use multilinear HSIC regression instead of OLS.",
    )
    rcd.add_argument(
        "--rcd-bw-method", choices=["mdbs", "scott", "silverman"], default="mdbs",
        help="HSIC kernel-bandwidth rule.",
    )
    rcd.add_argument(
        "--rcd-independence", choices=["hsic", "fcorr"], default="hsic",
        help="Independence test; fcorr is faster for large samples.",
    )
    rcd.add_argument(
        "--rcd-ind-corr", type=float, default=0.5,
        help="F-correlation cutoff when --rcd-independence fcorr is used.",
    )
    rcd.add_argument(
        "--rcd-directed-threshold", type=float, default=0.0,
        help="Optional absolute cutoff for finite directed coefficients.",
    )

    evaluation = parser.add_argument_group("Graph recovery and evaluation (all methods)")
    evaluation.add_argument(
        "--prune-threshold", type=_prune_threshold, default=0.05,
        help="Edge threshold: 'auto' for gap detection, or a float applied relative to max|B|.",
    )
    evaluation.add_argument("--prune-strategy", default="global", choices=["global", "greedy"])
    evaluation.add_argument(
        "--bidir-threshold", type=float, default=0.05,
        help="Correlation threshold for bidirected edges estimated from Z_hat.",
    )

    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _print_admg(directed, bidirected, label):
    n = directed.shape[0]
    header = "   " + " ".join(f"{j:2d}" for j in range(n))
    rows_dir   = [f"{i:2d} " + " ".join(f" {'1' if directed[i,j] else '.'}" for j in range(n)) for i in range(n)]
    rows_bidir = [f"{i:2d} " + " ".join(f" {'1' if bidirected[i,j] else '.'}" for j in range(n)) for i in range(n)]
    print(f"  {label}:")
    print(f"    directed       bidirected")
    print(f"    {header}      {header}")
    for rd, rb in zip(rows_dir, rows_bidir):
        print(f"    {rd}      {rb}")


def make_experiment_name(args, T, x_n):
    bow_free = "_bowfree" if args.bow_free_ground_truth else ""
    return (
        f"T{T}_n{args.n}_xn{x_n}_N{args.N}_"
        f"dense{args.graph_dense}_conf{args.confounder_dense}_"
        f"gtype{args.graph_type}_noise{args.noise_type}_"
        f"loss{args.loss_type}{bow_free}"
    )


def _dataset_run_name(args, T, x_n, seed, *, include_confounder=True):
    confounder = f"_conf{args.confounder_dense}" if include_confounder else ""
    bow_free = "_bowfree" if args.bow_free_ground_truth else ""
    return (
        f"n{args.n}_xn{x_n}_T{T}_N{args.N}_k{args.graph_dense}"
        f"{confounder}_{args.graph_type}_{args.noise_type}{bow_free}_rs{seed}"
    )


def _write_csv_with_header(path, header, row):
    """Append a row; write header first if file is new."""
    file_exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def _json_ready(value):
    """Convert NumPy-rich diagnostics into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(value), handle, indent=2, sort_keys=True)


def _generate_from_args(args, T, x_n):
    interv_dense = args.interv_dense
    if interv_dense is None:
        interv_dense = 1.0 / args.n if args.dgp == "hard_intervention" else 0.5
    return generate_dataset(
        args.dgp,
        T=T,
        n=args.n,
        xn=x_n,
        N=args.N,
        graph_dense=args.graph_dense,
        confounder_dense=args.confounder_dense,
        graph_type=args.graph_type,
        noise_type=args.noise_type,
        bow_free_ground_truth=args.bow_free_ground_truth,
        mask_dense=args.mask_dense,
        interv_dense=interv_dense,
    )




def _true_admg(dataset):
    return admg_reference_from_dataset(dataset)


def _run_admg_benchmark(args, baseline, T, x_n, model_dir, result_store):
    """Run/save/evaluate a baseline whose native output is an ADMG."""
    rows = []
    nested_metrics = []

    for seed in args.seeds:
        set_seed(seed)
        dataset = _generate_from_args(args, T, x_n)
        truth = _true_admg(dataset)
        save_dir = os.path.join(
            model_dir,
            _dataset_run_name(args, T, x_n, seed),
        )
        results_path = os.path.join(save_dir, "results.npz")

        if args.evaluate:
            if not os.path.exists(results_path):
                old_path = os.path.join(
                    model_dir,
                    _dataset_run_name(args, T, x_n, seed, include_confounder=False),
                    "results.npz",
                )
                if os.path.exists(old_path) and args.confounder_dense == 0.5:
                    results_path = old_path
                    save_dir = os.path.dirname(old_path)
                    print(f"Using pre-confounder-argument results from {results_path}")
                else:
                    raise FileNotFoundError(f"No saved ADMG results found in {results_path}")
            saved = np.load(results_path, allow_pickle=False)
            directed_weights = saved["directed_weights_hat"]
            bidirected = saved["bidirected_hat"]
            diagnostics = {}
            if args.method == "LiNGAM":
                from methods.lingam.baseline import recover_residuals

                bidirected_observations, base_index, base_reason = (
                    select_bidirected_observations(
                        dataset.domains,
                        dataset.domain_parameters,
                        args.dgp,
                    )
                )
                residuals = recover_residuals(
                    bidirected_observations,
                    directed_weights,
                )
                bidirected = estimate_bidirected_from_Z_hat_list(
                    residuals,
                    args.bidir_threshold,
                )
                diagnostics = {
                    "base_domain_index": base_index,
                    "base_domain_reason": base_reason,
                    "base_domain_samples": int(len(residuals)),
                    "bidirected_threshold": float(args.bidir_threshold),
                }
            estimate = ADMG(
                directed=saved["directed_hat"],
                bidirected=bidirected,
                directed_weights=directed_weights,
            )
            runtime_seconds = float(saved["runtime_seconds"].item())
            mixing = saved["A_hat_raw"]
        else:
            os.makedirs(save_dir, exist_ok=True)
            result = baseline.fit(dataset, args, T=T, n=args.n, xn=x_n)
            if not isinstance(result.graph, ADMG):
                raise TypeError(
                    f"The ADMG benchmark requires ADMG output, got {type(result.graph).__name__}."
                )
            estimate = result.graph
            runtime_seconds = result.runtime_seconds
            diagnostics = result.diagnostics
            mixing = result.artifacts["A_hat_raw"]
            save_values = {
                "directed_hat": estimate.directed,
                "bidirected_hat": estimate.bidirected,
                "directed_weights_hat": estimate.directed_weights,
                "directed_true": truth.directed,
                "bidirected_true": truth.bidirected,
                "directed_weights_true": truth.directed_weights,
                "A_hat_raw": mixing,
                "graph_kind": np.asarray(estimate.kind.value),
                "runtime_seconds": np.asarray(runtime_seconds),
            }
            raw_directed = result.artifacts.get("raw_directed_weights")
            if raw_directed is not None:
                save_values["raw_directed_weights"] = raw_directed
            omega_hat = result.artifacts.get("omega_hat")
            if omega_hat is not None:
                save_values["omega_hat"] = omega_hat
            raw_adjacency = result.artifacts.get("raw_adjacency")
            if raw_adjacency is not None:
                save_values["raw_adjacency"] = raw_adjacency
            np.savez(results_path, **save_values)

        metrics = evaluate_graph(truth, estimate)
        if args.method == "DCD" and not args.skip_dcd_pag:
            from evaluation.dcd_pag import (
                admg_to_pag,
                evaluate_dcd_pag_recovery,
                maximal_ancestral_projection,
            )

            truth_mag = maximal_ancestral_projection(truth)
            estimate_mag = maximal_ancestral_projection(estimate)
            truth_pag = admg_to_pag(truth)
            estimate_pag = admg_to_pag(estimate)
            metrics["pag_endpoint_recovery"] = evaluate_dcd_pag_recovery(
                truth,
                estimate,
                truth_pag=truth_pag,
                estimate_pag=estimate_pag,
            )
            result_store.write_array_artifact(
                seed=seed,
                save_dir=save_dir,
                filename="pag_results.npz",
                values={
                    "truth_mag_directed": truth_mag.directed,
                    "truth_mag_bidirected": truth_mag.bidirected,
                    "estimated_mag_directed": estimate_mag.directed,
                    "estimated_mag_bidirected": estimate_mag.bidirected,
                    "truth_pag_endpoints": truth_pag.endpoints,
                    "estimated_pag_endpoints": estimate_pag.endpoints,
                },
            )
        weighted_metrics = evaluate_graph(
            DirectedGraph(truth.directed_weights, weighted=True),
            DirectedGraph(estimate.directed_weights, weighted=True),
        )
        metrics["directed_weights"] = {
            "mse_all": weighted_metrics["mse_all"],
            "mse_edges": weighted_metrics["mse_edges"],
        }
        mcc, _, _ = compute_mcc(
            mixing,
            dataset.domains,
            dataset.latent_samples,
            args.n,
        )
        metrics["mcc"] = float(mcc)
        flat = flatten_metrics(metrics)
        row = {
            "seed": seed,
            "T": T,
            "n": args.n,
            "x_n": x_n,
            "N": args.N,
            "graph_dense": args.graph_dense,
            "confounder_dense": args.confounder_dense,
            "graph_type": args.graph_type,
            "noise_type": args.noise_type,
            "runtime_seconds": runtime_seconds,
            **flat,
        }
        rows.append(row)
        nested_metrics.append(metrics)
        result_store.write_metrics(
            seed=seed,
            save_dir=save_dir,
            metrics=metrics,
            diagnostics=diagnostics,
        )
        result_store.record_admg(
            seed=seed,
            save_dir=save_dir,
            num_domains=len(dataset.domains),
            truth=truth,
            estimate=estimate,
            metrics=metrics,
            runtime_seconds=runtime_seconds,
            mixing=mixing,
        )

        directed = metrics["directed"]
        bidirected = metrics["bidirected"]
        endpoint = metrics["endpoint_recovery"]
        _print_admg(truth.directed, truth.bidirected, "True ADMG")
        _print_admg(estimate.directed, estimate.bidirected, "Est. ADMG")
        print(
            f"Seed {seed} | MCC={mcc:.4f} | Directed SHD={directed['shd']} | "
            f"Prec={directed['precision']:.3f} | Rec={directed['recall']:.3f} | "
            f"F1={directed['f1']:.3f} | Bidir SHD={bidirected['shd']} | "
            f"Prec={bidirected['precision']:.3f} | Rec={bidirected['recall']:.3f} | "
            f"F1={bidirected['f1']:.3f} | runtime={runtime_seconds:.2f}s"
        )
        print(
            "  Exact-ADMG endpoint recovery | "
            f"Skeleton TPR={endpoint['skeleton']['tpr']:.3f} "
            f"FDR={endpoint['skeleton']['fdr']:.3f} | "
            f"Arrowhead TPR={endpoint['arrowhead']['tpr']:.3f} "
            f"FDR={endpoint['arrowhead']['fdr']:.3f} | "
            f"Tail TPR={endpoint['tail']['tpr']:.3f} "
            f"FDR={endpoint['tail']['fdr']:.3f}"
        )
        if args.method == "DCD" and not args.skip_dcd_pag:
            pag_endpoint = metrics["pag_endpoint_recovery"]
            print(
                "  DCD PAG recovery | "
                f"Skeleton TPR={pag_endpoint['skeleton']['tpr']:.3f} "
                f"FDR={pag_endpoint['skeleton']['fdr']:.3f} | "
                f"Arrowhead TPR={pag_endpoint['arrowhead']['tpr']:.3f} "
                f"FDR={pag_endpoint['arrowhead']['fdr']:.3f} | "
                f"Tail TPR={pag_endpoint['tail']['tpr']:.3f} "
                f"FDR={pag_endpoint['tail']['fdr']:.3f}"
            )

    metrics_path = os.path.join(model_dir, "graph_metrics.csv")
    header = list(rows[0])
    for row in rows:
        _write_csv_with_header(metrics_path, header, [row[key] for key in header])

    summary = summarize_metrics(nested_metrics)
    summary["mean.runtime_seconds"] = float(np.mean([row["runtime_seconds"] for row in rows]))
    _write_json(os.path.join(model_dir, "experiment_summary.json"), summary)
    print(
        f"Mean directed F1: {summary['mean.directed.f1']:.4f} | "
        f"Mean bidirected F1: {summary['mean.bidirected.f1']:.4f} | "
        f"Mean MCC: {summary['mean.mcc']:.4f}"
    )
    print(
        "Mean exact-ADMG endpoint recovery | "
        f"Skeleton TPR={summary['mean.endpoint_recovery.skeleton.tpr']:.4f} "
        f"FDR={summary['mean.endpoint_recovery.skeleton.fdr']:.4f} | "
        f"Arrowhead TPR={summary['mean.endpoint_recovery.arrowhead.tpr']:.4f} "
        f"FDR={summary['mean.endpoint_recovery.arrowhead.fdr']:.4f} | "
        f"Tail TPR={summary['mean.endpoint_recovery.tail.tpr']:.4f} "
        f"FDR={summary['mean.endpoint_recovery.tail.fdr']:.4f}"
    )
    if args.method == "DCD" and not args.skip_dcd_pag:
        print(
            "Mean DCD PAG recovery | "
            f"Skeleton TPR={summary['mean.pag_endpoint_recovery.skeleton.tpr']:.4f} "
            f"FDR={summary['mean.pag_endpoint_recovery.skeleton.fdr']:.4f} | "
            f"Arrowhead TPR={summary['mean.pag_endpoint_recovery.arrowhead.tpr']:.4f} "
            f"FDR={summary['mean.pag_endpoint_recovery.arrowhead.fdr']:.4f} | "
            f"Tail TPR={summary['mean.pag_endpoint_recovery.tail.tpr']:.4f} "
            f"FDR={summary['mean.pag_endpoint_recovery.tail.fdr']:.4f}"
        )


def main(args):
    x_n = int(args.x_n * args.n)
    T = 5 if args.dgp == "auxillary" else int(args.T * args.n)
    if args.method == "LiNGAM":
        run_family = (
            f'measure{args.lingam_measure}_dir{args.lingam_directed_threshold}_'
            f'bidir{args.bidir_threshold}_base{args.lingam_base_domain}_'
            f'random{args.lingam_random_state}'
        )
    elif args.method == "BANG":
        run_family = (
            f'K{args.bang_moment_degree}_level{args.bang_level}_'
            f'restrict{args.bang_restriction}_domains{args.bang_domain_mode}_'
            f'maxset{args.bang_max_set_size}_dir{args.bang_directed_threshold}'
        )
    elif args.method == "DCD":
        run_family = (
            f'class{args.dcd_admg_class}_lambda{args.dcd_lambda}_'
            f'threshold{args.dcd_threshold}_domains{args.dcd_domain_mode}'
            f'{"_standardizeTrue" if args.dcd_standardize else ""}_'
            f'restarts{args.dcd_num_restarts}_iter{args.dcd_max_iterations}_'
            f'cap{args.dcd_max_samples}_random{args.dcd_random_state}'
        )
    elif args.method == "RCD":
        run_family = (
            f'domains{args.rcd_domain_mode}_cap{args.rcd_max_samples}_'
            f'maxexp{args.rcd_max_explanatory_num}_cor{args.rcd_cor_alpha}_'
            f'ind{args.rcd_ind_alpha}_{args.rcd_independence}_'
            f'shapiro{args.rcd_shapiro_alpha}_mlhsicr{args.rcd_mlhsicr}_'
            f'dir{args.rcd_directed_threshold}'
        )
    else:
        run_family = (
            f'loss{args.loss_type}_lr{args.lr}_steps{args.num_steps}_init{args.num_initializations}'
        )
    legacy_model_dir = os.path.join(args.model_dir, run_family)
    model_dir = os.path.join(
        args.model_dir,
        args.dgp,
        args.method,
        run_family,
    )
    exp_name = make_experiment_name(args, T, x_n)
    os.makedirs(model_dir, exist_ok=True)
    result_store = RunResultStore(args, T=T, x_n=x_n)

    baseline = BASELINE_REGISTRY[args.method]

    if args.method in {"LiNGAM", "BANG", "DCD", "RCD"}:
        return _run_admg_benchmark(args, baseline, T, x_n, model_dir, result_store)

    mcc_scores = []
    all_metrics = []
    all_mse = []
    all_admg_metrics = []

    for seed in args.seeds:
        set_seed(seed)
        start_time = time.perf_counter()
        diagnostics = {}
        fit_runtime_seconds = None

        save_dir = os.path.join(
            model_dir,
            _dataset_run_name(args, T, x_n, seed),
        )
        os.makedirs(save_dir, exist_ok=True)
        results_path = os.path.join(save_dir, "results.npz")
        legacy_results_path = os.path.join(
            legacy_model_dir,
            _dataset_run_name(args, T, x_n, seed, include_confounder=False),
            "results.npz",
        )
        pre_confounder_results_path = os.path.join(
            model_dir,
            _dataset_run_name(args, T, x_n, seed, include_confounder=False),
            "results.npz",
        )

        dataset = _generate_from_args(args, T, x_n)
        X_list, A_true, B_true, Ds, sigma_vec, Z_list, skeleton = dataset.to_legacy_tuple()
        X_for_bidir, bidir_domain_index, _ = select_bidirected_observations(
            X_list,
            Ds,
            args.dgp,
        )
        
        bidirected_true = admg_reference_from_dataset(dataset).bidirected

        if args.evaluate:
            if not os.path.exists(results_path):
                if os.path.exists(pre_confounder_results_path) and args.confounder_dense == 0.5:
                    results_path = pre_confounder_results_path
                    print(f"Using pre-confounder-argument results from {results_path}")
                elif os.path.exists(legacy_results_path) and args.confounder_dense == 0.5:
                    results_path = legacy_results_path
                    print(f"Using legacy-layout results from {results_path}")
                else:
                    raise FileNotFoundError(
                        "No saved results found in either the namespaced or "
                        f"legacy location: {results_path}, {legacy_results_path}"
                    )
            data = np.load(results_path, allow_pickle=True)
            if "runtime_seconds" in data:
                fit_runtime_seconds = float(data["runtime_seconds"].item())
            A_hat_raw = data['A_hat']
            # Re-prune with current pruning args so evaluate mode can test different settings
            B_hat = find_P_and_prune_fast(
                np.linalg.inv(A_hat_raw),
                threshold=args.prune_threshold,
                relative_threshold=(args.prune_threshold != "auto"),
                strategy=args.prune_strategy,
                verbose=False,
            )
            # Recompute MCC from loaded A_hat_raw and freshly generated data
            maxcor, _, _ = compute_mcc(A_hat_raw, X_list, Z_list, args.n)
            print(f"Evaluated MCC: {maxcor:.4f} (saved was {data['MCC'].item():.4f})")
            z_hat_bidir = align_z_hat(
                X_for_bidir @ np.linalg.inv(A_hat_raw).T,
                A_hat_raw,
            )
            bidirected_hat = estimate_bidirected_from_Z_hat_list(
                z_hat_bidir,
                args.bidir_threshold,
            )
        else:
            baseline_result = baseline.fit(
                dataset,
                args,
                T=T,
                n=args.n,
                xn=x_n,
            )
            fit_runtime_seconds = baseline_result.runtime_seconds
            A_hat_raw = baseline_result.artifacts["A_hat_raw"]
            aux = baseline_result.diagnostics
            diagnostics = aux
            
            D_hats = aux["D_hats"]
            sigma_hat = aux["sigma_hat"]
            final_loss = aux["final_loss"]

            B_hat = baseline_result.graph.adjacency

            if aux.get("encoder_mcc") is not None:
                maxcor = aux["encoder_mcc"]
            else:
                maxcor, _, _ = compute_mcc(A_hat_raw, X_list, Z_list, args.n)
            encoder = aux.get("encoder")
            if encoder is not None:
                device = next(encoder.parameters()).device
                with torch.no_grad():
                    z_hat_bidir = encoder(torch.tensor(X_for_bidir, dtype=torch.float32, device=device)).cpu().numpy()
                z_hat_bidir = align_z_hat(z_hat_bidir, A_hat_raw)
                bidirected_hat = estimate_bidirected_from_Z_hat_list(z_hat_bidir, args.bidir_threshold)
                if bidir_domain_index is not None and Z_list is not None:
                    z_true_u = Z_list[bidir_domain_index]
                    corr_true = np.corrcoef(z_true_u.T)
                    corr_hat  = np.corrcoef(z_hat_bidir.T)
                    # cross-correlation: entry (i,j) = corr(z_true_i, z_hat_j)
                    cross = np.corrcoef(z_true_u.T, z_hat_bidir.T)
                    n_lat = z_true_u.shape[1]
                    cross_block = cross[:n_lat, n_lat:]
                    row_ind, col_ind = linear_sum_assignment(-np.abs(cross_block))
                    per_comp = cross_block[row_ind, col_ind]
                    np.set_printoptions(precision=3, suppress=True)
                    print("Corr matrix — true Z:\n", corr_true)
                    print("Corr matrix — Z_hat:\n",  corr_hat)
                    print("Cross-corr Z vs Z_hat (true rows, hat cols):\n", cross_block)
                    print("Per-component corr (optimal alignment):", np.round(per_comp, 3))
            else:
                z_hat_bidir = align_z_hat(
                    X_for_bidir @ np.linalg.inv(A_hat_raw).T,
                    A_hat_raw,
                )
                bidirected_hat = estimate_bidirected_from_Z_hat_list(
                    z_hat_bidir,
                    args.bidir_threshold,
                )

            np.savez(
                results_path,
                # estimated
                A_hat=A_hat_raw, B_hat=B_hat, D_hats=D_hats, sigma_hat=sigma_hat,
                # ground truth
                A_true=A_true, B_true=B_true, sigma_vec=sigma_vec,
                # Hard-intervention target sets have different lengths (the
                # reference set is empty), so NumPy 2.x cannot infer a regular
                # numeric array for them.
                Ds=np.asarray(Ds, dtype=object),
                # scalars
                MCC=maxcor, final_loss=final_loss,
                runtime_seconds=fit_runtime_seconds,
            )
            print(f"Results saved to {results_path}")

        directed_evaluation = evaluate_graph(
            DirectedGraph(A_true, weighted=True),
            DirectedGraph(B_hat, weighted=True),
        )
        metrics = {
            key: directed_evaluation[key]
            for key in ("tp", "fp", "fn", "precision", "recall", "f1", "shd", "reversed")
        }
        mse = {
            key: directed_evaluation[key]
            for key in ("mse_all", "mse_edges")
        }
        truth_admg = ADMG(
            directed=(np.abs(A_true) > 0).astype(int),
            bidirected=bidirected_true,
            directed_weights=A_true,
        )
        estimated_admg = ADMG(
            directed=(np.abs(B_hat) > 0).astype(int),
            bidirected=bidirected_hat,
            directed_weights=B_hat,
        )
        admg_metrics = evaluate_graph(truth_admg, estimated_admg)
        all_metrics.append(metrics)
        all_mse.append(mse)
        all_admg_metrics.append(admg_metrics)
        mcc_scores.append(maxcor)

        end_time = time.perf_counter()
        runtime_seconds = (
            fit_runtime_seconds
            if fit_runtime_seconds is not None
            else end_time - start_time
        )
        run_metrics = {
            **admg_metrics,
            "directed_weights": mse,
            "mcc": float(maxcor),
        }
        result_store.write_metrics(
            seed=seed,
            save_dir=save_dir,
            metrics=run_metrics,
            diagnostics=diagnostics,
        )
        result_store.record_admg(
            seed=seed,
            save_dir=save_dir,
            num_domains=len(dataset.domains),
            truth=truth_admg,
            estimate=estimated_admg,
            metrics=run_metrics,
            runtime_seconds=runtime_seconds,
            mixing=A_hat_raw,
        )
        directed_true_bin = (np.abs(A_true) > 0).astype(int)
        _print_admg(directed_true_bin, bidirected_true, "True ADMG")
        _print_admg((np.abs(B_hat) > 0).astype(int), bidirected_hat, "Est. ADMG")
        bm = admg_metrics["bidirected"]
        print(
            f"Seed {seed} | MCC={maxcor:.4f} | SHD={metrics['shd']} | "
            f"Prec={metrics['precision']:.3f} | Rec={metrics['recall']:.3f} | "
            f"F1={metrics['f1']:.3f} | Rev={metrics['reversed']} | "
            f"MSE_all={mse['mse_all']:.4f} | MSE_edges={mse['mse_edges']:.4f} | "
            f"Bidir Prec={bm['precision']:.3f} | Rec={bm['recall']:.3f} | "
            f"F1={bm['f1']:.3f} | SHD={bm['shd']} | "
            f"runtime: {runtime_seconds:.2f}s"
        )

    metrics_csv = os.path.join(model_dir, "graph_metrics.csv")
    for seed, metrics, mse, mcc, am in zip(args.seeds, all_metrics, all_mse, mcc_scores, all_admg_metrics):
        bm = am["bidirected"]
        _write_csv_with_header(
            metrics_csv,
            header=["seed", "T", "n", "x_n", "N", "graph_dense", "confounder_dense", "graph_type", "noise_type",
                    "tp", "fp", "fn", "precision", "recall", "f1", "shd", "reversed",
                    "mse_all", "mse_edges", "mcc",
                    "bidir_tp", "bidir_fp", "bidir_fn",
                    "bidir_precision", "bidir_recall", "bidir_f1", "bidir_shd"],
            row=[seed, T, args.n, x_n, args.N, args.graph_dense, args.confounder_dense, args.graph_type, args.noise_type,
                 metrics["tp"], metrics["fp"], metrics["fn"],
                 metrics["precision"], metrics["recall"], metrics["f1"],
                 metrics["shd"], metrics["reversed"],
                 mse["mse_all"], mse["mse_edges"], mcc,
                 bm["tp"], bm["fp"], bm["fn"],
                 bm["precision"], bm["recall"], bm["f1"], bm["shd"]],
        )

    # Experiment summary CSV
    mean_mcc       = np.mean(mcc_scores)
    std_mcc        = np.std(mcc_scores)
    mean_precision = np.mean([m["precision"] for m in all_metrics])
    mean_recall    = np.mean([m["recall"]    for m in all_metrics])
    mean_f1        = np.mean([m["f1"]        for m in all_metrics])
    mean_shd       = np.mean([m["shd"]       for m in all_metrics])
    mean_rev       = np.mean([m["reversed"]  for m in all_metrics])
    mean_mse_all   = np.mean([m["mse_all"]   for m in all_mse])
    mean_mse_edges = np.nanmean([m["mse_edges"] for m in all_mse])
    mean_bidir_prec = np.mean([m["bidirected"]["precision"] for m in all_admg_metrics])
    mean_bidir_rec  = np.mean([m["bidirected"]["recall"]    for m in all_admg_metrics])
    mean_bidir_f1   = np.mean([m["bidirected"]["f1"]        for m in all_admg_metrics])
    mean_bidir_shd  = np.mean([m["bidirected"]["shd"]       for m in all_admg_metrics])

    _write_csv_with_header(
        os.path.join(model_dir, "experiment_summary.csv"),
        header=["T", "n", "x_n", "N", "graph_dense", "confounder_dense", "graph_type", "noise_type",
                "loss_type", "lr", "num_steps", "num_initializations",
                "scheduler", "grad_clip", "patience",
                "prune_threshold", "prune_strategy", "bidir_threshold",
                "mean_mcc", "std_mcc", "mean_precision", "mean_recall",
                "mean_f1", "mean_shd", "mean_reversed",
                "mean_mse_all", "mean_mse_edges",
                "mean_bidir_precision", "mean_bidir_recall", "mean_bidir_f1", "mean_bidir_shd"],
        row=[T, args.n, x_n, args.N, args.graph_dense, args.confounder_dense, args.graph_type, args.noise_type,
             args.loss_type, args.lr, args.num_steps, args.num_initializations,
             args.scheduler, args.grad_clip, args.patience,
             args.prune_threshold, args.prune_strategy, args.bidir_threshold,
             mean_mcc, std_mcc, mean_precision, mean_recall, mean_f1, mean_shd, mean_rev,
             mean_mse_all, mean_mse_edges,
             mean_bidir_prec, mean_bidir_rec, mean_bidir_f1, mean_bidir_shd],
    )

    print(f"Mean MCC: {mean_mcc:.4f} ± {std_mcc:.4f}")
    print(f"Mean Precision: {mean_precision:.4f} | Mean Recall: {mean_recall:.4f} | Mean F1: {mean_f1:.4f}")
    print(f"Mean SHD: {mean_shd:.4f} | Mean Reversed: {mean_rev:.4f}")
    print(f"Mean MSE (all): {mean_mse_all:.4f} | Mean MSE (true edges): {mean_mse_edges:.4f}")
    print(f"Bidir Precision: {mean_bidir_prec:.4f} | Recall: {mean_bidir_rec:.4f} | F1: {mean_bidir_f1:.4f} | SHD: {mean_bidir_shd:.4f}")


if __name__ == "__main__":
    main(build_parser().parse_args())
