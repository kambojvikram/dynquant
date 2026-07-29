"""The ``dynquant`` command line.

Subcommands are added as their phase lands; this module only ever wires up the
ones that exist. Two properties are worth keeping as it grows:

**Import stays cheap.** Each handler is imported inside its own function, so
``dynquant doctor`` on a machine with a broken CUDA install does not fail while
importing an allocator it was never going to run. ``argparse`` builds the whole
parser tree at startup, so parser construction must not import anything either.

**Exit codes are meaningful.** ``0`` success, ``1`` a DynQuant-level failure with
a diagnosed message, ``2`` argparse usage error (argparse's own convention),
``130`` interrupted. Scripts and CI can branch on that; a traceback is only shown
for genuinely unexpected exceptions, where the traceback is the useful part.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ._version import __version__
from .constants import BIT_OPTIONS, COMPUTE_DTYPES, DEFAULT_GROUP_SIZE

if TYPE_CHECKING:
    # `_SubParsersAction` is generic to type checkers but a plain class at runtime,
    # so subscripting it outside an annotation raises TypeError. The alias lives
    # under TYPE_CHECKING and the annotations are strings (`from __future__` above),
    # which is what makes the parameterised form legal here.
    _SubParsers = argparse._SubParsersAction[argparse.ArgumentParser]

__all__ = ["build_parser", "main"]

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INTERRUPTED = 130


# --------------------------------------------------------------------------
# Flag groups shared by more than one subcommand
# --------------------------------------------------------------------------
#
# Shared because the commands must agree, not to save typing. `inspect` and
# `quantize` allocate from the same inputs -- if `--group-size` meant something
# different to each of them, inspecting a map would tell you nothing about the map
# you were about to write.


def _add_compute_device(parser: argparse.ArgumentParser) -> None:
    """For commands that encode weights, as opposed to merely reading their shapes.

    Separate from ``--device`` on purpose: that one says where the model lives, and
    the two answers differ exactly when it matters most. A model too large for VRAM
    is loaded onto the CPU and its encoding still belongs on the GPU.
    """
    parser.add_argument(
        "--compute-device",
        default="auto",
        metavar="DEVICE",
        help=(
            "where the encoding arithmetic runs, independent of --device: 'auto' "
            "(default) uses CUDA when present, 'none' keeps it on whichever device "
            "holds the weights, or name one. Weights move one at a time, so this "
            "costs one tensor of VRAM and not a second copy of the model"
        ),
    )


def _add_loading(parser: argparse.ArgumentParser, *, device: str = "cuda") -> None:
    parser.add_argument(
        "--device",
        default=device,
        help=f"device to load onto (default: {device})",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=COMPUTE_DTYPES,
        help="compute dtype to load in (default: bfloat16)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="execute modelling code shipped with the checkpoint",
    )


def _add_allocation(parser: argparse.ArgumentParser) -> None:
    """Where the widths come from, and how they are priced."""
    parser.add_argument(
        "--stats",
        help="signal file or directory written by DynQuantCallback during fine-tuning",
    )
    parser.add_argument(
        "--moments",
        help=(
            "per-channel second moments, which switch the allocator from rank-product "
            "ordering to measured sensitivity. Wants a GPU: every tensor is quantized "
            "at every candidate width"
        ),
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=DEFAULT_GROUP_SIZE,
        help=f"quantization group size along the input dimension (default: {DEFAULT_GROUP_SIZE})",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="encode without an offset (worse for weights, kept for comparison)",
    )
    parser.add_argument(
        "--hard-floors",
        action="store_true",
        help=(
            "refuse to breach a role's floor. The budget then may be unreachable, and "
            "the allocator says so rather than silently returning the floor map"
        ),
    )
    parser.add_argument(
        "--role",
        action="append",
        metavar="NAME=ROLE",
        help="override the classified role of one module; repeatable",
    )


def _add_map_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--map",
        help="a bit map written earlier by `dynquant inspect --save-map`",
    )
    parser.add_argument(
        "--map-key",
        help="which map inside the file to use, when it holds more than one",
    )


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def _cmd_doctor(args: argparse.Namespace) -> int:
    from .doctor import diagnose, render

    report = diagnose()
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, default=str))
    else:
        print(render(report))
    # A non-zero exit on a failed check is the whole point of having this in CI:
    # `dynquant doctor` in a Dockerfile turns a silently-degraded image into a
    # build failure.
    return EXIT_OK if report.ok else EXIT_FAILED


def _add_doctor(subparsers: _SubParsers) -> None:
    parser = subparsers.add_parser(
        "doctor",
        help="check the installation and verify it numerically",
        description=(
            "Report the environment, the selected backend and why the others were "
            "rejected, then run a numerical self-check. Exits non-zero if any check "
            "found something that would make results untrustworthy."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of the text report",
    )
    parser.set_defaults(handler=_cmd_doctor)


# --------------------------------------------------------------------------
# version
# --------------------------------------------------------------------------


def _cmd_version(args: argparse.Namespace) -> int:
    from ._version import CHECKPOINT_FORMAT_VERSION, KERNEL_ABI_VERSION, STATS_SCHEMA_VERSION
    from .constants import KERNELS_IMPORT_NAME

    try:
        kernels = __import__(KERNELS_IMPORT_NAME).__version__
    except Exception:  # noqa: BLE001 - not installed is the common case
        kernels = "not installed"

    payload = {
        "dynquant-core": __version__,
        "dynquant-kernels": kernels,
        "kernel_abi": KERNEL_ABI_VERSION,
        "checkpoint_format": CHECKPOINT_FORMAT_VERSION,
        "stats_schema": STATS_SCHEMA_VERSION,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        width = max(len(key) for key in payload)
        for key, value in payload.items():
            print(f"{key:<{width}}  {value}")
    return EXIT_OK


def _add_version(subparsers: _SubParsers) -> None:
    parser = subparsers.add_parser(
        "version",
        help="print package versions and format contract numbers",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.set_defaults(handler=_cmd_version)


# --------------------------------------------------------------------------
# inspect
# --------------------------------------------------------------------------


def _cmd_inspect(args: argparse.Namespace) -> int:
    from .commands import inspect

    return inspect.run(args)


def _add_inspect(subparsers: _SubParsers) -> None:
    parser = subparsers.add_parser(
        "inspect",
        help="classify a model, score it, and report what an allocation would do",
        description=(
            "Read the module tree, score every quantizable module from the "
            "fine-tuning signals, and report the allocation at one or more budgets "
            "without touching a weight. Reports within-role concordance of width with "
            "score, every floor violation, and every module the score never saw -- the "
            "three things a width histogram cannot show."
        ),
    )
    parser.add_argument("model", help="checkpoint directory or Hub id")
    # CPU by default: only names and shapes are read, and a second copy of the
    # weights on the GPU during an evaluation turns an analysis into an OOM in the
    # run that matters.
    _add_loading(parser, device="cpu")
    _add_allocation(parser)
    parser.add_argument(
        "--target",
        type=float,
        nargs="+",
        metavar="BITS",
        help="average bits per weight; repeatable, so several budgets are compared",
    )
    parser.add_argument("--target-size", metavar="SIZE", help="a size budget, e.g. 6.5GiB")
    parser.add_argument(
        "--target-ratio", type=float, metavar="R", help="a fraction of the fp16 size"
    )
    parser.add_argument(
        "--uniform",
        type=int,
        nargs="+",
        choices=BIT_OPTIONS,
        metavar="BITS",
        help="also report a uniform map at these widths, as the control arm",
    )
    parser.add_argument(
        "--narrowest",
        type=int,
        default=8,
        help="how many of the narrowest modules to list per target (default: 8)",
    )
    parser.add_argument(
        "--save-map", metavar="PATH", help="write the bit maps for `quantize --map`"
    )
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="exit non-zero if any floor was breached, for use in CI",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of the text report")
    parser.set_defaults(handler=_cmd_inspect)


# --------------------------------------------------------------------------
# quantize
# --------------------------------------------------------------------------


def _cmd_quantize(args: argparse.Namespace) -> int:
    from .commands import quantize

    return quantize.run(args)


def _add_quantize(subparsers: _SubParsers) -> None:
    parser = subparsers.add_parser(
        "quantize",
        help="allocate widths and encode the weights",
        description=(
            "Write a model whose every value has been quantized to its allocated "
            "width and decoded back. Loadable by plain transformers with no DynQuant "
            "installed, and any accuracy measured on it is the quantized model's "
            "accuracy -- but it is stored in the compute dtype, so its size on disk "
            "is not the quantized size. The packed size is printed alongside it, and "
            "`dynquant eval --map` is what measures it in VRAM."
        ),
    )
    parser.add_argument("model", help="checkpoint directory or Hub id")
    parser.add_argument("-o", "--output", help="directory to write (required unless --dry-run)")
    _add_loading(parser)
    _add_compute_device(parser)
    _add_allocation(parser)
    _add_map_input(parser)
    parser.add_argument("--target", type=float, metavar="BITS", help="average bits per weight")
    parser.add_argument("--target-size", metavar="SIZE", help="a size budget, e.g. 6.5GiB")
    parser.add_argument(
        "--target-ratio", type=float, metavar="R", help="a fraction of the fp16 size"
    )
    parser.add_argument(
        "--uniform",
        type=int,
        choices=BIT_OPTIONS,
        metavar="BITS",
        help="every module at one width, structural roles excepted",
    )
    parser.add_argument("--tokenizer", help="where to copy the tokenizer from (default: the model)")
    parser.add_argument(
        "--save-map", metavar="PATH", help="under --dry-run, write the map that would be applied"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="allocate and report, then stop before touching a weight",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-module progress")
    parser.add_argument("--json", action="store_true", help="also emit a JSON summary")
    parser.set_defaults(handler=_cmd_quantize)


# --------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------


def _cmd_eval(args: argparse.Namespace) -> int:
    from .commands import evaluate

    return evaluate.run(args)


def _add_eval(subparsers: _SubParsers) -> None:
    parser = subparsers.add_parser(
        "eval",
        help="score a model on a task, and compare two scores honestly",
        description=(
            "Greedy decode, a fixed few-shot prefix drawn by seed from a split that is "
            "not scored, and the same scorer whatever produced the text. --out writes "
            "the per-problem correctness vector, and --compare runs a paired McNemar "
            "test against an earlier record -- refusing to compare across differing "
            "task, split, shots, seed or limit."
        ),
    )
    parser.add_argument("model", help="checkpoint directory or Hub id")
    parser.add_argument(
        "--task",
        required=True,
        choices=("gsm8k", "casehold", "banking77"),
        help="which task to score",
    )
    _add_loading(parser)
    _add_compute_device(parser)
    _add_map_input(parser)
    parser.add_argument(
        "--group-size",
        type=int,
        default=DEFAULT_GROUP_SIZE,
        help=f"group size to pack --map at, if the file does not say (default: {DEFAULT_GROUP_SIZE})",
    )
    parser.add_argument("--split", default="test", help="split to score (default: test)")
    parser.add_argument(
        "--shots", type=int, help="few-shot examples in the prompt (default: per task)"
    )
    parser.add_argument(
        "--shot-split", default="train", help="split the shots are drawn from (default: train)"
    )
    parser.add_argument("--shot-seed", type=int, default=0, help="seed for the shots (default: 0)")
    parser.add_argument("--limit", type=int, help="score only the first N problems")
    parser.add_argument("--batch-size", type=int, help="generation batch size (default: per task)")
    parser.add_argument("--max-new-tokens", type=int, help="decode budget (default: per task)")
    parser.add_argument(
        "--max-prompt-tokens", type=int, help="truncate prompts to this (default: per task)"
    )
    parser.add_argument("--tokenizer", help="where to load the tokenizer from (default: the model)")
    parser.add_argument("--label", help="name for this run in the output (default: task:model)")
    parser.add_argument(
        "--keep-predictions",
        type=int,
        default=0,
        metavar="N",
        help=(
            "record the first N generations verbatim. A count rather than a flag: the "
            "full set is hundreds of KB of JSON per run, and a handful is what is "
            "actually read when a score looks wrong"
        ),
    )
    parser.add_argument("--out", metavar="FILE", help="write the record, including the hit vector")
    parser.add_argument(
        "--compare", metavar="FILE", help="paired McNemar test against a record from --out"
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        metavar="ACC",
        help="exit non-zero below this accuracy, for use in CI",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress")
    parser.add_argument("--json", action="store_true", help="also emit the record as JSON")
    parser.set_defaults(handler=_cmd_eval)


# --------------------------------------------------------------------------
# bench
# --------------------------------------------------------------------------


def _cmd_bench(args: argparse.Namespace) -> int:
    from .commands import bench

    return bench.run(args)


def _add_bench(subparsers: _SubParsers) -> None:
    parser = subparsers.add_parser(
        "bench",
        help="measure the packed GEMV against this card's achievable bandwidth",
        description=(
            "Decode is memory-bound, so the figure that matters is the fraction of "
            "achievable read bandwidth the packed kernel reaches. Timing comes from "
            "kernel-level profiler events, not wall clock, and the denominator is "
            "measured on this card rather than taken from the datasheet."
        ),
    )
    parser.add_argument(
        "--model",
        help="take the distinct Linear shapes from this model (read on the meta device)",
    )
    parser.add_argument(
        "--shape",
        action="append",
        metavar="[LABEL=]OUTxIN",
        help="measure one rectangle; repeatable, and combinable with --model",
    )
    parser.add_argument(
        "--max-shapes",
        type=int,
        default=8,
        help="how many distinct model shapes to measure, largest first (default: 8)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=list(BIT_OPTIONS),
        choices=BIT_OPTIONS,
        metavar="BITS",
        help=f"widths to measure (default: {' '.join(str(b) for b in BIT_OPTIONS)})",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=DEFAULT_GROUP_SIZE,
        help=f"quantization group size (default: {DEFAULT_GROUP_SIZE})",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1,
        help="activation rows per call; 1 is decode (default: 1)",
    )
    parser.add_argument(
        "--iters", type=int, help="timed iterations per measurement (default: by shape size)"
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=COMPUTE_DTYPES,
        help="activation and baseline dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="execute modelling code shipped with the checkpoint, under --model",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help=(
            "benchmark the torch fallback deliberately. Without this, a run without "
            "the compiled kernels is refused rather than reported as a kernel number"
        ),
    )
    parser.add_argument("--out", metavar="FILE", help="write the measurements as JSON")
    parser.add_argument("--quiet", action="store_true", help="suppress per-shape progress")
    parser.add_argument("--json", action="store_true", help="also emit the measurements as JSON")
    parser.set_defaults(handler=_cmd_bench)


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynquant",
        description="Training-dynamics-driven mixed-precision LLM quantization.",
    )
    parser.add_argument("-V", "--version", action="version", version=f"dynquant {__version__}")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="-v for INFO, -vv for DEBUG (overrides $DYNQUANT_LOG_LEVEL)",
    )
    # required=True so a bare `dynquant` prints usage and exits 2 rather than
    # succeeding silently, which reads as "it worked".
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    _add_inspect(subparsers)
    _add_quantize(subparsers)
    _add_eval(subparsers)
    _add_bench(subparsers)
    _add_doctor(subparsers)
    _add_version(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from .errors import DynQuantError

    args = build_parser().parse_args(argv)

    if args.verbose:
        from ._logging import enable_logging

        enable_logging("DEBUG" if args.verbose > 1 else "INFO")

    try:
        return int(args.handler(args))
    except DynQuantError as exc:
        # Every DynQuantError is written to be read by whoever hit it, so the
        # message is the whole error report -- a traceback would only bury it.
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
