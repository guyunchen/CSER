"""
PlotNeuralNet diagram for the best UnifiedSERModel configuration.

Usage:
  1. Copy this file to the PlotNeuralNet project root, or keep it here and
     update PLOT_NEURAL_NET_ROOT below.
  2. Run:
       python scripts/plot_unified_ser_arch.py
  3. Compile the generated TeX:
       bash ../tikzmake.sh plot_unified_ser_arch

The diagram matches:
  logmel_80 -> IdentityFrontend -> TemporalConvEncoder -> 3 x DA-LS4 residual
  blocks -> multi-head statistical temporal attention -> regression head.
"""

import os
import sys


PLOT_NEURAL_NET_ROOT = os.environ.get("PLOT_NEURAL_NET_ROOT", "..")
sys.path.append(PLOT_NEURAL_NET_ROOT)

from pycore.tikzeng import *  # noqa: F401,F403


def to_note(name, text, offset, to, width=3.2, height=0.9):
    return r"""
\node[canvas is zy plane at x=0, text width=%0.1fcm, align=center]
  (%s) at ($(%s) + (%s)$)
  {\small %s};
""" % (width, name, to, offset, text)


def to_sequence_box(
    name,
    caption,
    xlabel,
    zlabel,
    offset,
    to="(0,0,0)",
    width=2.0,
    height=20,
    depth=46,
    fill=r"\ConvColor",
):
    return r"""
\pic[shift={%s}] at %s
  {Box={
    name=%s,
    caption=%s,
    xlabel={{%s}},
    zlabel=%s,
    fill=%s,
    height=%s,
    width=%s,
    depth=%s
  }};
""" % (offset, to, name, caption, xlabel, zlabel, fill, height, width, depth)


def to_dals4_block(name, idx, offset, to):
    caption = r"DA-LS4 Block %d" % idx
    return [
        to_sequence_box(
            name=name,
            caption=caption,
            xlabel=r"256",
            zlabel=r"$T/4$",
            offset=offset,
            to=to,
            width=2.8,
            height=28,
            depth=42,
            fill=r"\ConvReluColor",
        ),
        to_note(
            name=f"{name}_note",
            text=(
                r"$d_{state}=32$, $p=3$\\"
                r"$A_{dyn}$ gate + $\lambda$ MLP\\"
                r"GELU + AddNorm + Dropout 0.35"
            ),
            offset="(0,-3.0,0)",
            to=f"{name}-center",
            width=4.1,
        ),
    ]


def to_residual_skip(name, src, dst, yshift=1.25):
    return r"""
\draw [copyconnection]
  (%s-west) -- ++(0,%0.2f,0) -- node {\copymidarrow}
  ($(%s-east) + (0,%0.2f,0)$) -- (%s-east);
""" % (src, yshift, dst, yshift, dst)


def main():
    arch = [
        to_head(PLOT_NEURAL_NET_ROOT),
        to_cor(),
        to_begin(),
        to_sequence_box(
            name="input",
            caption=r"Log-Mel Features",
            xlabel=r"80",
            zlabel=r"$T$",
            offset="(0,0,0)",
            to="(0,0,0)",
            width=1.4,
            height=20,
            depth=52,
            fill=r"\ConvColor",
        ),
        to_note(
            name="input_note",
            text=r"Input: $B \times T \times 80$\\fixed dB normalization",
            offset="(0,-3.0,0)",
            to="input-center",
        ),
        to_sequence_box(
            name="frontend",
            caption=r"Identity Frontend",
            xlabel=r"80",
            zlabel=r"$T$",
            offset="(2.4,0,0)",
            to="(input-east)",
            width=1.2,
            height=20,
            depth=52,
            fill=r"\SoftmaxColor",
        ),
        to_connection("input", "frontend"),
        to_sequence_box(
            name="conv1",
            caption=r"Conv1D Encoder",
            xlabel=r"256",
            zlabel=r"$T/2$",
            offset="(2.4,0,0)",
            to="(frontend-east)",
            width=2.0,
            height=28,
            depth=46,
            fill=r"\ConvColor",
        ),
        to_note(
            name="conv1_note",
            text=r"Conv1d $k=5$, stride 2\\GroupNorm + GELU + Dropout",
            offset="(0,-3.0,0)",
            to="conv1-center",
        ),
        to_connection("frontend", "conv1"),
        to_sequence_box(
            name="conv2",
            caption=r"Depthwise-Separable Conv",
            xlabel=r"256",
            zlabel=r"$T/4$",
            offset="(2.5,0,0)",
            to="(conv1-east)",
            width=2.0,
            height=28,
            depth=40,
            fill=r"\ConvColor",
        ),
        to_note(
            name="conv2_note",
            text=r"Depthwise $k=3$, stride 2\\Pointwise $1\times1$ + LayerNorm",
            offset="(0,-3.0,0)",
            to="conv2-center",
        ),
        to_connection("conv1", "conv2"),
        *to_dals4_block("dals4_1", 1, "(2.7,0,0)", "(conv2-east)"),
        to_connection("conv2", "dals4_1"),
        *to_dals4_block("dals4_2", 2, "(2.7,0,0)", "(dals4_1-east)"),
        to_connection("dals4_1", "dals4_2"),
        *to_dals4_block("dals4_3", 3, "(2.7,0,0)", "(dals4_2-east)"),
        to_connection("dals4_2", "dals4_3"),
        to_residual_skip("skip1", "dals4_1", "dals4_1"),
        to_residual_skip("skip2", "dals4_2", "dals4_2"),
        to_residual_skip("skip3", "dals4_3", "dals4_3"),
        to_sequence_box(
            name="attn",
            caption=r"Temporal Attention Pooling",
            xlabel=r"256",
            zlabel=r"1",
            offset="(3.0,0,0)",
            to="(dals4_3-east)",
            width=2.1,
            height=22,
            depth=22,
            fill=r"\PoolColor",
        ),
        to_note(
            name="attn_note",
            text=(
                r"4-head attention scores\\"
                r"weighted mean + weighted std\\"
                r"Linear: $512 \rightarrow 256$"
            ),
            offset="(0,-3.0,0)",
            to="attn-center",
            width=4.0,
        ),
        to_connection("dals4_3", "attn"),
        to_sequence_box(
            name="head1",
            caption=r"Regression MLP",
            xlabel=r"128",
            zlabel=r"1",
            offset="(2.7,0,0)",
            to="(attn-east)",
            width=1.6,
            height=16,
            depth=16,
            fill=r"\FcColor",
        ),
        to_note(
            name="head_note",
            text=r"Linear $256\rightarrow128$\\ReLU + Dropout 0.35",
            offset="(0,-3.0,0)",
            to="head1-center",
        ),
        to_connection("attn", "head1"),
        to_sequence_box(
            name="out",
            caption=r"Emotion Output",
            xlabel=r"3",
            zlabel=r"1",
            offset="(2.0,0,0)",
            to="(head1-east)",
            width=1.0,
            height=10,
            depth=10,
            fill=r"\SoftmaxColor",
        ),
        to_note(
            name="out_note",
            text=r"Linear $128\rightarrow3$\\continuous emotion regression",
            offset="(0,-3.0,0)",
            to="out-center",
            width=3.3,
        ),
        to_connection("head1", "out"),
        to_note(
            name="title",
            text=(
                r"\Large UnifiedSERModel Best Configuration\\"
                r"\small input\_dim=80, hidden\_dim=256, layers=3, "
                r"DA-LS4($d_{state}=32,p=3$), dropout=0.35"
            ),
            offset="(7.5,3.2,0)",
            to="input-center",
            width=14.5,
        ),
        to_end(),
    ]

    to_generate(arch, "plot_unified_ser_arch.tex")


if __name__ == "__main__":
    main()
