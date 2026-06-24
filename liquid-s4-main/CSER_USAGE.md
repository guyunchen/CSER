# CSER Usage

`liquid-s4-main` is kept as a vendored copy of the author's Liquid-S4 source tree.
CSER uses it only for the A4 `original_ls4` baseline.

Most files in this directory are not used by CSER training. They belong to the
author repository's standalone experiments, configs, datasets, callbacks,
generation scripts, and optional CUDA extensions.

## Entry Point Used By CSER

CSER imports:

```text
src.models.sequence.ss.s4.S4
```

through:

```text
models/original_ls4.py
```

The normal Lite-GLSER, DA-LS4, S4, and CFC experiments do not instantiate this
adapter and do not need `liquid-s4-main`.

## Files Loaded For Original-LS4

A runtime import trace for `OriginalLS4Layer(d_model=8, d_state=4, p_order=2)`
loads these files:

```text
src/models/functional/cauchy.py
src/models/functional/krylov.py
src/models/functional/toeplitz.py
src/models/functional/vandermonde.py
src/models/hippo/hippo.py
src/models/nn/__init__.py
src/models/nn/components.py
src/models/nn/exprnn/orthogonal.py
src/models/nn/exprnn/parametrization.py
src/models/nn/residual.py
src/models/sequence/__init__.py
src/models/sequence/base.py
src/models/sequence/block.py
src/models/sequence/ff.py
src/models/sequence/model.py
src/models/sequence/pool.py
src/models/sequence/ss/dplr.py
src/models/sequence/ss/kernel.py
src/models/sequence/ss/s4.py
src/models/sequence/unet.py
src/utils/config.py
src/utils/registry.py
```

The optional Cauchy CUDA extension is not required. If it is missing, the author
implementation falls back to a slower kernel.

## Cleanup Policy

Do not delete individual files from `liquid-s4-main` unless you are deliberately
creating a minimal vendored subset and validating A4 again. Keeping the full
source tree preserves the original license, provenance, and import behavior.
