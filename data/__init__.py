"""Data-generating processes and their shared typed containers."""

from data.types import GeneratedDataset, GroundTruth

# Keep package import lightweight. The registry imports every DGP (and their
# optional dependencies), so runners should import it from data.registry.
__all__ = ["GeneratedDataset", "GroundTruth"]
