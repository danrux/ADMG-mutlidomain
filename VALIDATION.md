# Validation

Validated on 6 October 2026 with a fresh Python 3.11.8 environment on Windows
and the packages pinned in requirements-lock.txt (PyTorch 2.10.0+cpu).

Passed after the naming and single-example revision:

- Installing the lean synthetic dependency lock in an isolated environment.
- Parsing all Python files; importing every retained source module; checking
  all package exports and CLI method choices.
- Confirming there is exactly one example file and no paper YAML configurations,
  campaign planners, sweep scripts, plot/table generators, or study CLI flags.
- Running that example with every synthetic method: MuDo-nll, SP, diagGMM, UMNI,
  VaDE-linear, LiNGAM, BANG, DCD, and RCD.
- Checking saved metrics, graph arrays, and the renamed method labels.
- Reloading saved SP and diagGMM models with --evaluate; writing evaluation
  outputs separately; confirming fitted graph files remain unchanged.
- Generating finite sample arrays for all five synthetic DGPs, including nonlinear
  masking, and evaluating their ground-truth graphs against themselves.
- Running the causalAssembly linear option with diagGMM and MuDo-nll, including
  official graph loading, generation, fitting and graph evaluation; confirming
  both methods used the same generated-data hash.

The causalAssembly linear check used causalAssembly 1.2.1 and requests 2.34.2
without installing its optional R bridge. Nonlinear DRF generation and the full
R/rpy2/drf installation were not run; their modules import successfully and their
numerical generator implementation is retained. The example uses small budgets
and is not an evaluation of final scientific performance.

Temporary environments, check scripts, generated arrays and logs were removed
after verification. The original development repository was not edited.
