"""Command-line entry point for multi-domain causal discovery benchmarks."""

from experiments.run_benchmark import (
    DGP_REGISTRY,
    METHOD_REGISTRY,
    _print_admg,
    _prune_threshold,
    _write_csv_with_header,
    build_parser,
    main,
    make_experiment_name,
    set_seed,
)

__all__ = [
    "DGP_REGISTRY",
    "METHOD_REGISTRY",
    "_print_admg",
    "_prune_threshold",
    "_write_csv_with_header",
    "build_parser",
    "main",
    "make_experiment_name",
    "set_seed",
]


if __name__ == "__main__":
    main(build_parser().parse_args())
