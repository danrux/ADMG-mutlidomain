# Multi-domain causal discovery

Methods, synthetic and causalAssembly data generation, graph representations,
and graph evaluation, with one small end-to-end example.

## Installation

Use Python 3.11 and Git. From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate`.
For CPU PyTorch wheels, add
`--extra-index-url https://download.pytorch.org/whl/cpu` to the install command.
`requirements.txt` pins both direct and transitive dependencies for reproducible
installation. SP uses Cooper's legacy API at the pinned source commit.

## One end-to-end example

```text
python examples/quickstart.py
```

The default is MuDo-nll on scaling data: three observed variables, seven domains,
200 samples per domain, seed 2, and 30 optimization steps. It generates data,
fits the model, scores directed/bidirected edges, and saves the results under
`outputs/quickstart/synthetic/MuDo-nll/`. The example uses a small training
budget for quick execution.

The same example accepts `--method` and `--output`. Available methods and the
synthetic data selected for each are:

| Method | Example data |
| --- | --- |
| MuDo-nll | Scaling |
| SP | Masking |
| UMNI | Hard intervention |
| diagGMM | Auxiliary Gaussian mixture |
| VaDE-linear | Auxiliary Gaussian mixture |
| LiNGAM, BANG, DCD, RCD | Scaling |

Each run saves `metrics.json`, `graphs.npz`, and `run_config.json`, together
with model-specific artifacts. Evaluation reports directed and bidirected F1
and SHD separately; DCD also reports PAG endpoint recovery. The renamed method
modules are `methods/SP.py` and `methods/diagGMM.py`.

The CLI `main.py` exposes the same method names under `--baseline` / `--method`;
use `python main.py --help` to inspect data and optimization settings.

## causalAssembly support

The same example has `--dataset causalassembly` for a small linear benchmark
using the official Station-1/2 topology. This mode supports MuDo-nll and diagGMM
and writes generated arrays as `dataset.npz`. Initial graph loading requires
network access. Install the additional dependencies first:

```text
python -m pip install -r requirements-causalassembly.txt
```

The upstream package declares an R bridge dependency, so configure R before
installing the full additional requirements. Set `R_HOME` if R is not found;
on Windows the pinned bridge uses `RPY2_CFFI_MODE=ABI`.
Nonlinear generation remains available as a library in
`experiments/causalassembly_admg/nonlinear.py`; it also needs the R package
`drf`, installed with `R -e "install.packages('drf', repos='https://cloud.r-project.org')"`.
The official [causalAssembly documentation](https://github.com/boschresearch/causalAssembly)
describes its source data and R requirements.

## Layout

- `methods/`: estimators and baseline adapters.
- `data/`: synthetic generators, including nonlinear masking.
- `graphs/`, `evaluation/`, `evaluate.py`, `utils.py`: graph and numerical utilities.
- `experiments/`: single-run execution, per-run result saving, and causalAssembly libraries.
- `examples/quickstart.py`: the single example, with selectable method/data.

Third-party notices are preserved in `THIRD_PARTY_NOTICES.md` and `third_party/`.
