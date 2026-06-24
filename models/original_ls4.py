"""Adapter for the author's Liquid-S4 implementation.

CSER keeps the paper baseline separate from the local DA-LS4 implementation.
Only the A4 `original_ls4` experiment instantiates this module. The import is
lazy so normal Lite-GLSER/S4/CFC runs do not need the author source tree.
"""

import importlib
import logging
import os
import sys
import types
from pathlib import Path

import torch.nn as nn

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


AUTHOR_REPO_DIR = "liquid-s4-main"
AUTHOR_S4_IMPORT = "src.models.sequence.ss.s4"


AUTHOR_FILES_LOADED_FOR_CSER = (
    "src/models/functional/cauchy.py",
    "src/models/functional/krylov.py",
    "src/models/functional/toeplitz.py",
    "src/models/functional/vandermonde.py",
    "src/models/hippo/hippo.py",
    "src/models/nn/__init__.py",
    "src/models/nn/components.py",
    "src/models/nn/exprnn/orthogonal.py",
    "src/models/nn/exprnn/parametrization.py",
    "src/models/nn/residual.py",
    "src/models/sequence/__init__.py",
    "src/models/sequence/base.py",
    "src/models/sequence/block.py",
    "src/models/sequence/ff.py",
    "src/models/sequence/model.py",
    "src/models/sequence/pool.py",
    "src/models/sequence/ss/dplr.py",
    "src/models/sequence/ss/kernel.py",
    "src/models/sequence/ss/s4.py",
    "src/models/sequence/unet.py",
    "src/utils/config.py",
    "src/utils/registry.py",
)


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _prepend_path(path):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)


def _author_root():
    root = _repo_root() / AUTHOR_REPO_DIR
    if not root.exists():
        raise ImportError(
            f"Missing author Liquid-S4 source tree: {root}. "
            "Original-LS4 is only needed for the A4 baseline."
        )
    return root


def _ensure_dependency_paths(author_root):
    local_deps = _repo_root() / ".codex_deps"
    if local_deps.exists():
        _prepend_path(local_deps)
    _prepend_path(author_root)


def _install_train_logger_stub(author_root):
    """Avoid importing the author's full training stack for a single S4 layer."""
    train_utils = types.ModuleType("src.utils.train")
    train_utils.get_logger = lambda name=__name__, level=logging.INFO: logging.getLogger(name)

    utils_pkg = types.ModuleType("src.utils")
    utils_pkg.__path__ = [str(author_root / "src" / "utils")]
    utils_pkg.train = train_utils

    sys.modules["src.utils"] = utils_pkg
    sys.modules["src.utils.train"] = train_utils

    src_pkg = importlib.import_module("src")
    setattr(src_pkg, "utils", utils_pkg)


def load_author_s4_class():
    author_root = _author_root()
    _ensure_dependency_paths(author_root)
    _install_train_logger_stub(author_root)

    try:
        module = importlib.import_module(AUTHOR_S4_IMPORT)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Cannot import the author Liquid-S4 S4 layer. Install any missing "
            "dependencies from liquid-s4-main/requirements.txt before running "
            "the A4 Original-LS4 baseline."
        ) from exc
    return module.S4


class OriginalLS4Layer(nn.Module):
    """Project-local wrapper around the author Liquid-S4 `S4` layer.

    The author implementation can use `(B, H, L)` or `(B, L, H)` depending on
    `transposed`; CSER uses `(B, L, H)`, so this adapter fixes `transposed=False`
    and unwraps the `(output, next_state)` tuple.
    """

    def __init__(
        self,
        d_model,
        d_state=64,
        p_order=2,
        liquid_kernel="polyb",
        dropout=0.0,
        l_max=None,
        allcombs=True,
        lcontract=None,
        **kernel_args,
    ):
        super().__init__()
        if p_order <= 1:
            raise ValueError(f"Original Liquid-S4 requires p_order/liquid_degree >= 2, got {p_order}.")

        s4_cls = load_author_s4_class()
        self.layer = s4_cls(
            d_model=d_model,
            d_state=d_state,
            l_max=l_max,
            channels=1,
            bidirectional=False,
            activation="gelu",
            postact="glu",
            initializer=None,
            weight_norm=False,
            hyper_act=None,
            dropout=dropout,
            tie_dropout=False,
            bottleneck=None,
            gate=None,
            transposed=False,
            verbose=False,
            shift=False,
            linear=False,
            liquid_kernel=liquid_kernel,
            liquid_degree=p_order,
            allcombs=allcombs,
            lcontract=lcontract,
            mode="nplr",
            measure="legs",
            rank=1,
            dt_min=0.001,
            dt_max=0.1,
            lr={"dt": 0.001, "A": 0.001, "B": 0.001},
            n_ssm=1,
            deterministic=False,
            **kernel_args,
        )

    def forward(self, u, lengths=None):
        y, _ = self.layer(u, lengths=lengths)
        return y
