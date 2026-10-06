# RCD provenance

This baseline wraps `lingam.RCD`, the maintained MIT-licensed implementation
provided by the LiNGAM project:
https://github.com/cdt15/lingam

The method is described in:

Takashi Nicholas Maeda and Shohei Shimizu, *RCD: Repetitive causal discovery of
linear non-Gaussian acyclic models with latent confounders*, AISTATS 2020,
PMLR 108:735--745.

The adapter only converts the implementation's native adjacency convention—NaN
pairs for shared latent confounders and finite entries for directed effects—to
the repository's canonical ADMG representation.
