"""Graph-only intervention target and regime design."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from math import comb
from typing import Any

from .config import InterventionConfig
from .graph import GraphSplit


@dataclass(frozen=True)
class Regime:
    index: int
    targets: tuple[str, ...]
    strength: str


@dataclass(frozen=True)
class InterventionDesign:
    selected_targets: tuple[str, ...]
    observed_descendants: dict[str, tuple[str, ...]]
    regimes: tuple[Regime, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_targets": list(self.selected_targets),
            "observed_descendants": {
                node: list(descendants)
                for node, descendants in self.observed_descendants.items()
            },
            "regimes": [asdict(regime) for regime in self.regimes],
        }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def select_intervention_targets(
    graph,
    split: GraphSplit,
    n_targets: int,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Greedily cover diverse observed descendant sets using graph structure only."""
    import networkx as nx

    observed = set(split.observed_nodes)
    descendants = {
        node: tuple(sorted(set(nx.descendants(graph, node)) & observed))
        for node in split.hidden_nodes
    }
    candidates = [node for node, values in descendants.items() if values]
    candidates.sort(key=lambda node: (-len(descendants[node]), node))
    if not candidates:
        raise ValueError("No hidden node has an observed descendant")
    selected: list[str] = []
    covered: set[str] = set()
    target_count = min(n_targets, len(candidates))
    while candidates and len(selected) < target_count:
        def score(node: str):
            values = set(descendants[node])
            maximum_overlap = max(
                (_jaccard(values, set(descendants[old])) for old in selected),
                default=0.0,
            )
            return (
                len(values - covered),
                int(len(values) >= 2),
                1.0 - maximum_overlap,
                len(values),
                tuple(-ord(character) for character in node),
            )

        chosen = max(candidates, key=score)
        selected.append(chosen)
        covered.update(descendants[chosen])
        candidates.remove(chosen)
    return tuple(selected), {node: descendants[node] for node in selected}


def build_intervention_design(
    graph,
    split: GraphSplit,
    config: InterventionConfig,
    n_regimes: int,
) -> InterventionDesign:
    """Build deterministic hidden-only regimes under the configured policy."""
    if config.target_policy == "all_hidden":
        import networkx as nx

        observed = set(split.observed_nodes)
        targets = tuple(split.hidden_nodes)
        descendants = {
            node: tuple(sorted(set(nx.descendants(graph, node)) & observed))
            for node in targets
        }
        regimes = [Regime(index=0, targets=(), strength="observational")]
        for regime_index in range(1, n_regimes):
            strength = config.strength_cycle[
                (regime_index - 1) % len(config.strength_cycle)
            ]
            regimes.append(Regime(regime_index, targets, strength))
        assert all(set(regime.targets) <= set(split.hidden_nodes) for regime in regimes)
        return InterventionDesign(targets, descendants, tuple(regimes))

    capacity = (n_regimes - 1) * config.max_targets_per_regime
    targets, descendants = select_intervention_targets(
        graph, split, min(config.n_targets, capacity)
    )
    if not targets:
        raise ValueError("At least one intervention target is required")
    if len(targets) < config.min_targets_per_regime:
        raise ValueError(
            f"Only {len(targets)} eligible targets were selected, fewer than "
            f"min_targets_per_regime={config.min_targets_per_regime}"
        )
    if config.unique_target_sets:
        maximum_size = min(config.max_targets_per_regime, len(targets))
        minimum_size = min(config.min_targets_per_regime, maximum_size)
        sizes = tuple(range(minimum_size, maximum_size + 1))
        n_available = sum(comb(len(targets), size) for size in sizes)
        if n_regimes - 1 > n_available:
            raise ValueError(
                f"Requested {n_regimes - 1} unique sparse intervention sets, "
                f"but only {n_available} subsets of size {minimum_size}..{maximum_size} exist"
            )
        pools = {
            size: list(combinations(targets, size))
            for size in sizes
        }
        selected_sets: list[tuple[str, ...]] = []
        usage = {node: 0 for node in targets}
        for offset in range(n_regimes - 1):
            preferred_size = sizes[offset % len(sizes)]
            available_sizes = [
                size
                for step in range(len(sizes))
                if pools[size := sizes[(sizes.index(preferred_size) + step) % len(sizes)]]
            ]
            size = available_sizes[0]

            def subset_score(values: tuple[str, ...]):
                value_set = set(values)
                maximum_overlap = max(
                    (_jaccard(value_set, set(old)) for old in selected_sets),
                    default=0.0,
                )
                return (
                    -sum(usage[node] for node in values),
                    -max((usage[node] for node in values), default=0),
                    -maximum_overlap,
                    tuple(-targets.index(node) for node in values),
                )

            chosen = max(pools[size], key=subset_score)
            pools[size].remove(chosen)
            selected_sets.append(chosen)
            for node in chosen:
                usage[node] += 1

        regimes = [Regime(index=0, targets=(), strength="observational")]
        for regime_index, chosen in enumerate(selected_sets, start=1):
            strength = config.strength_cycle[
                (regime_index - 1) % len(config.strength_cycle)
            ]
            regimes.append(Regime(regime_index, chosen, strength))
        assert len({regime.targets for regime in regimes[1:]}) == n_regimes - 1
        assert {node for regime in regimes for node in regime.targets} == set(targets)
        return InterventionDesign(targets, descendants, tuple(regimes))

    regimes = [Regime(index=0, targets=(), strength="observational")]
    next_uncovered = 0
    for regime_index in range(1, n_regimes):
        size = min(
            config.min_targets_per_regime
            + (regime_index - 1)
            % (config.max_targets_per_regime - config.min_targets_per_regime + 1),
            len(targets),
        )
        chosen = []
        while len(chosen) < size:
            candidate = targets[next_uncovered % len(targets)]
            next_uncovered += 1
            if candidate not in chosen:
                chosen.append(candidate)
        strength = config.strength_cycle[(regime_index - 1) % len(config.strength_cycle)]
        regimes.append(Regime(regime_index, tuple(chosen), strength))
    used = {node for regime in regimes for node in regime.targets}
    if used != set(targets):
        # Increase target-set sizes deterministically when regimes are fewer than targets.
        missing = [node for node in targets if node not in used]
        mutable = list(regimes)
        for offset, node in enumerate(missing):
            index = 1 + offset % (n_regimes - 1)
            values = list(mutable[index].targets)
            if len(values) >= config.max_targets_per_regime:
                values[-1] = node
            else:
                values.append(node)
            mutable[index] = Regime(index, tuple(dict.fromkeys(values)), mutable[index].strength)
        regimes = mutable
        used = {node for regime in regimes for node in regime.targets}
        if used != set(targets):
            raise AssertionError("Regime construction failed to cover every selected target")
    hidden = set(split.hidden_nodes)
    assert all(set(regime.targets) <= hidden for regime in regimes)
    assert all(descendants[node] for node in targets)
    return InterventionDesign(targets, descendants, tuple(regimes))
