# BANG provenance

This implementation is a Python port of Algorithms 1--4 from:

Y. Samuel Wang and Mathias Drton, *Causal Discovery with Unobserved Confounding
and Non-Gaussian Data*, JMLR 24 (2023), arXiv:2007.11131.

It follows the authors' `ngBap::bang` reference implementation:
https://github.com/ysamwang/ngBap/blob/main/R/bang.R

The `ngBap` package declares the MIT license. This port preserves the matrix
convention `B[v, u] != 0` for the directed edge `u -> v` and implements the
empirical-likelihood moment-test variant in Python.
