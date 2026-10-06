"""Configuration objects for the causalAssembly benchmark suite."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any



@dataclass(frozen=True)
class SplitConfig:
    # Keep the original graph-selected Station-2 prefix as the default.  The
    # station1_only ablation exposes every Station-2 node and hides Station 1.
    mode: str = "station2_prefix"
    hidden_prefix_fraction: float = 0.5
    minimum_fraction: float = 0.3
    maximum_fraction: float = 0.7
    minimum_observed: int = 4
    minimum_observed_edges: int = 2
    minimum_hidden_to_observed_edges: int = 2
    minimum_bidirected_edges: int = 2
    minimum_eligible_targets: int = 3

    def __post_init__(self) -> None:
        if self.mode not in {"station2_prefix", "station1_only"}:
            raise ValueError("split.mode must be 'station2_prefix' or 'station1_only'")


@dataclass(frozen=True)
class InterventionConfig:
    target_policy: str = "diverse"
    mechanism: str = "replace"
    variance_multiplier_distribution: str = "fixed_strength"
    variance_multiplier_min: float = 0.5
    variance_multiplier_max: float = 1.5
    # Optional mild index-dependent scale change for regimes that repeatedly
    # intervene on the same eligible hidden variable.
    regime_scale_trend: float = 0.0
    unique_target_sets: bool = False
    n_targets: int = 6
    min_targets_per_regime: int = 1
    max_targets_per_regime: int = 3
    strength_cycle: tuple[str, ...] = ("weak", "medium", "strong")
    mean_shift_multipliers: dict[str, float] = field(
        default_factory=lambda: {"weak": 0.35, "medium": 0.7, "strong": 1.1}
    )
    scale_multipliers: dict[str, float] = field(
        default_factory=lambda: {"weak": 1.1, "medium": 1.3, "strong": 1.6}
    )

    def __post_init__(self) -> None:
        if self.target_policy not in {"diverse", "all_hidden"}:
            raise ValueError("target_policy must be 'diverse' or 'all_hidden'")
        if self.mechanism not in {"replace", "noise_variance"}:
            raise ValueError("mechanism must be 'replace' or 'noise_variance'")
        if self.variance_multiplier_distribution not in {"fixed_strength", "uniform"}:
            raise ValueError("variance_multiplier_distribution must be 'fixed_strength' or 'uniform'")
        if self.variance_multiplier_distribution == "uniform" and self.mechanism != "noise_variance":
            raise ValueError("Uniform variance multipliers require noise_variance interventions")
        if not 0 < self.variance_multiplier_min < self.variance_multiplier_max:
            raise ValueError("Uniform variance multiplier bounds must be positive and increasing")
        if not 0 <= self.regime_scale_trend < 1:
            raise ValueError("regime_scale_trend must lie in [0, 1)")
        if not 1 <= self.min_targets_per_regime <= self.max_targets_per_regime:
            raise ValueError("Intervention target-set sizes must satisfy 1 <= min <= max")


@dataclass(frozen=True)
class MethodConfig:
    num_steps: int = 5000
    lr: float = 1e-3
    batch_size: int = 6144
    num_initializations: int = 1
    loss_type: str = "nll"
    scheduler: str = "cosine"
    grad_clip: float = 1.0
    patience: int = 2000
    checkpoint_policy: str = "best"
    validate_normalization: bool = True
    min_normalizing_diagonal: float = 0.0
    exact_components: int | None = None
    exact_jitter: float = 1e-6
    log_every: int = 250


@dataclass(frozen=True)
class ExperimentConfig:
    setup: str
    output_dir: str = "experiments/causalassembly_admg/results"
    n_regimes: int = 9
    n_per_regime: int = 5000
    n_heldout_per_regime: int = 0
    n_repetitions: int = 10
    base_seed: int = 2222
    alpha: float = 0.05
    bidirected_min_abs_correlation: float = 0.05
    oracle_bidirected_thresholds: tuple[float, ...] = (0.02, 0.03, 0.05)
    dependence_sample_source: str = "training"
    require_more_regimes_than_observed: bool = False
    directed_threshold: float | str = 0.05
    directed_postprocessing: str = "dag_pruned"
    prune_strategy: str = "global"
    coefficient_min: float = 0.4
    coefficient_max: float = 0.8
    noise_distribution: str = "gaussian"
    noise_scale: float = 1.0
    drf_fit_samples: int | None = None
    drf_smoothed_sources: bool = True
    split: SplitConfig = field(default_factory=SplitConfig)
    interventions: InterventionConfig = field(default_factory=InterventionConfig)
    methods: dict[str, MethodConfig] = field(
        default_factory=lambda: {
            "diagGMM": MethodConfig(),
            "MuDo-nll": MethodConfig(lr=1e-2),
        }
    )

    def __post_init__(self) -> None:
        if self.setup not in {"linear", "nonlinear"}:
            raise ValueError("setup must be 'linear' or 'nonlinear'")
        if self.n_regimes < 2:
            raise ValueError("n_regimes includes observation and must be at least two")
        if self.n_per_regime < 2 or self.n_heldout_per_regime < 0:
            raise ValueError("training size must be at least two and held-out size nonnegative")
        if self.dependence_sample_source not in {"training", "heldout"}:
            raise ValueError("dependence_sample_source must be 'training' or 'heldout'")
        if self.dependence_sample_source == "heldout" and self.n_heldout_per_regime < 2:
            raise ValueError("held-out dependence testing requires at least two held-out samples")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        if self.directed_postprocessing not in {"dag_pruned", "threshold_only"}:
            raise ValueError("directed_postprocessing must be 'dag_pruned' or 'threshold_only'")
        if any(not 0 <= value < 1 for value in self.oracle_bidirected_thresholds):
            raise ValueError("oracle bidirected thresholds must lie in [0, 1)")

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        # Keep default generation fingerprints stable for frozen datasets.
        if self.split.mode == "station2_prefix":
            values["split"].pop("mode")
        if self.interventions.regime_scale_trend == 0.0:
            values["interventions"].pop("regime_scale_trend")
        return values




