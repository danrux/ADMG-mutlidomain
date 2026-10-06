"""Registry and typed adapters for the repository's data generators."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from data.linear_auxillary import generate_data as auxillary_generate_data
from data.linear_hard_intervention import generate_data as hard_interv_generate_data
from data.linear_mask import generate_data as mask_generate_data
from data.linear_scaling import generate_data as linear_generate_data
from data.nonlinear_mask import generate_data as nonlinear_mask_generate_data
from data.types import GeneratedDataset


DGP_REGISTRY: dict[str, Callable[..., tuple[Any, ...]]] = {
    "scaling": linear_generate_data,
    "mask": mask_generate_data,
    "nonlinear_mask": nonlinear_mask_generate_data,
    "hard_intervention": hard_interv_generate_data,
    "auxillary": auxillary_generate_data,
}


def generate_dataset(name: str, **kwargs: Any) -> GeneratedDataset:
    """Run an existing DGP and adapt its unchanged tuple result."""
    try:
        generator = DGP_REGISTRY[name]
    except KeyError as exc:
        choices = ", ".join(sorted(DGP_REGISTRY))
        raise ValueError(f"Unknown DGP {name!r}; choose one of: {choices}") from exc

    legacy_result = generator(**kwargs)
    return GeneratedDataset.from_legacy_tuple(
        legacy_result,
        metadata={"dgp": name, "generator_kwargs": dict(kwargs)},
    )
