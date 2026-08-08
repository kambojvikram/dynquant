"""Seven arms on one fine-tuned checkpoint, at two byte budgets that the baselines chose.

The panel is bf16, GPTQ 4b/3b, AWQ 4b/3b, DynQuant 4b/3b. Every quantized arm sees the
same fine-tuned LFM2.5-8B-A1B, is scored on the same decontaminated text-to-SQL test set,
and stores per-item hits so the comparisons are paired.

Who sets the budget, and why it is not DynQuant
-----------------------------------------------

GPTQ and AWQ do not take a size. They take a width, and the bytes fall out of what
``compressed-tensors`` writes: the payload at ``bits``, an fp16 scale per group, and a
zero point packed at the weight's own width -- ``bits + 16`` per group of 128, so 4.1565
bits per parameter at 4 bits and 3.1488 at 3.

DynQuant does take a size, and its own format is more expensive per group: an fp16 scale
*and* an fp16 offset, 32 bits per group rather than 20, so its uniform 4-bit arm is 4.25
bits per parameter and its uniform 3-bit arm is 3.25. Anchoring the panel on DynQuant's
uniform arms would therefore hand DynQuant **2.3% more bytes at 4 bits and 3.2% more at
3** than the baselines spend -- a size advantage sitting inside the arm whose accuracy is
the claim.

So the anchors are the baselines' numbers and DynQuant is pinned to them. The overhead its
format pays comes out of its own payload, which is the point: a method that stores more
metadata has fewer bits left for weights at the same footprint, and that is a real cost
rather than an accounting convention. ``--target-size`` accepts a bare byte count, so the
pin is exact rather than rounded through a unit.

Each accounting describes the format that arm actually writes. Charging both by DynQuant's
rules would overstate the baselines; charging both by the baselines' would let DynQuant
write metadata it never paid for.

Run::

    python experiments/phase4/arms_lfm2.py plan --model runs/s4/.../merged
    python experiments/phase4/arms_lfm2.py run --model runs/s4/.../merged \
        --stats runs/s4/.../dynquant_stats.json --out runs/s4/arms --limit 400
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines_lfm2 import accounted_bytes

#: 4 and 3 because those are the two regimes phase 2 separated, and because
#: ``compressed-tensors`` packs 4 bits natively -- the 4-bit arms can be saved and pushed,
#: the 3-bit ones are scored in process.
ANCHOR_WIDTHS = (4, 3)

#: How far an arm's accounted size may sit from its anchor before the comparison stops
#: being about assignments. The same number as the phase-3 panel, for the same reason.
MATCH_TOLERANCE = 0.001


@dataclass(slots=True)
class Arm:
    """One row of the panel: what to run, at which budget, under which label."""

    label: str
    kind: str  # "ceiling" | "gptq" | "awq" | "dq"
    anchor: int | None
    target_bytes: int | None
    nbytes: int | None = None
    record: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def anchor_bytes(model: str, group_size: int) -> dict[int, int]:
    """The byte budget each width implies, read off the baselines' own format rules.

    Computed against a meta-device copy of the architecture, so it costs no weights and no
    GPU -- see :func:`baselines_lfm2.accounted_bytes` for why it must not be measured off
    the model llm-compressor hands back.
    """
    return {
        width: int(accounted_bytes(model, width, group_size)["accounted_bytes"])
        for width in ANCHOR_WIDTHS
    }


def plan_arms(budgets: dict[int, int]) -> list[Arm]:
    """The panel, in the order it should run.

    The ceiling first: it is the only arm that can fail for a reason that is not about
    quantization, and finding that out after six quantization passes wastes all six. Then
    both widths of each method rather than both methods of each width, so a method that
    cannot be built at all is discovered on its first arm.
    """
    arms = [Arm(label="bf16", kind="ceiling", anchor=None, target_bytes=None)]
    for width in ANCHOR_WIDTHS:
        budget = budgets[width]
        for kind in ("gptq", "awq", "dq"):
            arms.append(Arm(label=f"{kind}_{width}b", kind=kind, anchor=width, target_bytes=budget))
    return arms


def check_matched(arm: Arm) -> None:
    """Refuse an arm whose realised size drifted off the budget it was given.

    ``--target-size`` is a ceiling, and an allocator that cannot spend the last few bits
    lands under it. Two arms a percent apart are not a comparison of assignments; the
    larger one has a size advantage that will be read as accuracy. This is what makes "at
    matched bytes" a fact about the run rather than a sentence in the report.
    """
    if arm.target_bytes is None or arm.nbytes is None:
        return
    drift = abs(arm.nbytes - arm.target_bytes) / arm.target_bytes
    print(
        f"  {arm.label}: {arm.nbytes - arm.target_bytes:+d} B off the {arm.anchor}-bit "
        f"anchor ({drift:.5%})",
        flush=True,
    )
    if drift > MATCH_TOLERANCE:
        raise SystemExit(
            f"{arm.label} is {drift:.3%} off its anchor ({arm.nbytes} vs "
            f"{arm.target_bytes} bytes), over the {MATCH_TOLERANCE:.3%} tolerance. These "
            f"arms are not byte-matched and any accuracy difference between them is "
            f"confounded with size."
        )


def eval_flags(args: argparse.Namespace, label: str) -> list[str]:
    """The scoring contract every arm shares.

    One builder rather than one per arm kind: the shot count, the shot seed, the prompt
    style and the decode budget are what make two records comparable, and a panel where
    one arm was scored under different ones is not a panel.
    """
    flags = [
        "--label",
        label,
        # On every arm, including the baselines, because a panel where one arm was loaded
        # somewhere else is not a panel either. It is a passthrough rather than a constant
        # so the whole driver can be rehearsed end to end on a machine whose GPU is busy --
        # which is the machine this panel is scheduled on, and the seven-arm rehearsal is
        # the only check that reaches the manifest, the pairing guard and the table.
        "--device",
        args.device,
        "--split",
        args.split,
        "--shots",
        str(args.shots),
        "--shot-seed",
        str(args.shot_seed),
        "--prompt-style",
        args.prompt_style,
        # Stated on every command rather than left to the task spec, which would answer
        # 320 and censor the arms. Unconditional, unlike `--limit`: an absent budget is
        # only safe when every arm inherits the same default, and the defaults differ
        # between the CLI parser and the in-process config -- so the panel says it.
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--keep-predictions",
        str(args.keep_predictions),
    ]
    for name, value in (("--limit", args.limit), ("--batch-size", args.batch_size)):
        if value is not None:
            flags += [name, str(value)]
    return flags


def baseline_cmd(args: argparse.Namespace, arm: Arm, out: Path) -> list[str]:
    """``baselines_lfm2 run`` for one GPTQ or AWQ arm.

    A subprocess per arm, not a loop in one process: llm-compressor leaves observer state
    and Hessians on the model it calibrated, and the cheapest way to be certain none of it
    reaches the next arm is for the next arm to be a new interpreter.
    """
    return [
        sys.executable,
        str(Path(__file__).resolve().parent / "baselines_lfm2.py"),
        "run",
        "--model",
        args.model,
        "--method",
        arm.kind,
        "--bits",
        str(arm.anchor),
        "--group-size",
        str(args.group_size),
        "--calib-samples",
        str(args.calib_samples),
        "--seq-len",
        str(args.seq_len),
        "--seed",
        str(args.seed),
        "--out",
        str(out),
        *eval_flags(args, arm.label),
    ]


def dq_inspect_cmd(args: argparse.Namespace, arm: Arm, save_map: Path) -> list[str]:
    """Allocate a DynQuant arm against the baselines' byte count.

    ``-m dynquant`` on ``sys.executable`` rather than the console script, so the subprocess
    is this environment's package and not whichever ``dynquant`` is first on ``PATH``.
    """
    cmd = [
        sys.executable,
        "-m",
        "dynquant",
        "inspect",
        args.model,
        "--stats",
        args.stats,
        "--group-size",
        str(args.group_size),
        "--target-size",
        str(arm.target_bytes),
        "--save-map",
        str(save_map),
        "--json",
    ]
    if args.moments:
        cmd += ["--moments", args.moments]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


def dq_eval_cmd(args: argparse.Namespace, arm: Arm, save_map: Path, out: Path) -> list[str]:
    """Score a DynQuant arm through its map rather than through a written checkpoint.

    ``dynquant quantize`` would write a bf16-decoded copy -- same numerics, 16.9 GB of disk
    per arm, and a folder whose size does not mean what its name says. ``eval --map`` applies
    the same arithmetic in memory without the copy.

    ``--map-apply encode`` and not the default ``pack``, because of what this model is. The
    packed runtime replaces ``Linear`` and ``Embedding`` *modules*, and 91.5% of this
    checkpoint's parameters are batched expert banks -- bare 3-D parameters with no module to
    replace. Encoding runs the identical encoder at the identical widths and writes the
    reconstruction back in bf16, so the accuracy is the one being measured; what it does not
    do is hold the packed size, and the panel never claims it does. Every byte figure here
    comes from the map, which was priced by the allocator and is checked against the
    baselines' anchor before a single problem is scored.
    """
    cmd = [
        sys.executable,
        "-m",
        "dynquant",
        "eval",
        args.model,
        "--task",
        "text2sql",
        "--map",
        str(save_map),
        "--map-key",
        str(arm.target_bytes),
        "--map-apply",
        "encode",
        "--group-size",
        str(args.group_size),
        "--out",
        str(out),
        *eval_flags(args, arm.label),
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


def ceiling_cmd(args: argparse.Namespace, arm: Arm, out: Path) -> list[str]:
    """The unquantized reference: the same eval, no map, no recipe."""
    cmd = [
        sys.executable,
        "-m",
        "dynquant",
        "eval",
        args.model,
        "--task",
        "text2sql",
        "--out",
        str(out),
        *eval_flags(args, arm.label),
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


def _run(cmd: list[str], *, what: str) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode:
        raise SystemExit(f"{what} failed with exit code {result.returncode}")


def map_nbytes(save_map: Path, key: str) -> int:
    """The allocated size of one map, from the file the allocator wrote.

    Read back rather than taken from the request: ``--target-size`` is what was asked for
    and this is what was allocated, and :func:`check_matched` exists precisely because
    those two are not the same number.
    """
    if not save_map.is_file():
        raise SystemExit(
            f"{save_map} does not exist. Under --resume this means the record was kept and "
            f"the allocation that produced it was not, so the arm's realised size cannot be "
            f"checked against its anchor. Delete the record and let the arm run again."
        )
    payload = json.loads(save_map.read_text(encoding="utf-8"))
    maps = payload["maps"]
    if key not in maps:
        raise SystemExit(
            f"{save_map} holds {sorted(maps)} but not {key!r}. The allocation ran under a "
            f"different budget than this arm was planned at."
        )
    return int(maps[key]["nbytes"])


def do_plan(args: argparse.Namespace) -> int:
    """The budgets and the panel, with no weights loaded and no GPU touched."""
    budgets = anchor_bytes(args.model, args.group_size)
    arms = plan_arms(budgets)
    print(
        json.dumps(
            {
                "anchors": {
                    str(width): {
                        "bytes": budget,
                        "gib": round(budget / 2**30, 4),
                        "source": "gptq/awq under compressed-tensors' own rules",
                    }
                    for width, budget in budgets.items()
                },
                "arms": [
                    {
                        "label": arm.label,
                        "kind": arm.kind,
                        "anchor": arm.anchor,
                        "target_bytes": arm.target_bytes,
                    }
                    for arm in arms
                ],
            },
            indent=2,
        )
    )
    return 0


def require_one_stack() -> None:
    """Every arm runs under the interpreter this script was started with. Check it can.

    ``sys.executable`` is used for all seven subprocesses on purpose. The baselines need
    llm-compressor and the DynQuant arms need only dynquant-core, so it is tempting to run
    each under whichever environment has it -- and then the panel compares accuracies
    produced by two different transformers versions, which is a difference in the
    measuring instrument reported as a difference between methods.

    llm-compressor is the constraint, so the panel runs wherever it lives. Checked before
    the ceiling arm rather than at the first baseline: the ceiling is an hour of
    generation, and discovering the environment is wrong after it is an hour wasted.
    """
    if importlib.util.find_spec("llmcompressor") is None:
        raise SystemExit(
            f"llmcompressor is not importable from {sys.executable}, and the GPTQ and AWQ "
            f"arms need it. Run the whole panel under the interpreter that has it -- not "
            f"just those arms, or the baselines and the DynQuant arms are scored by "
            f"different stacks."
        )


def check_uncensored(record: Path) -> None:
    """The ceiling closed its reasoning inside the budget, so the roof is the model's.

    Called on the ceiling alone and before anything else runs. A quantized arm that
    deliberates past the cap is measuring a real effect of quantization and is recorded as
    such; a *ceiling* that does is measuring the cap, and every arm underneath it is then
    compared beneath a lowered roof -- differences between them come partly from how much
    budget each had left, and the whole panel is scored against a headline that is not the
    model's.

    Strict rather than tolerant. This fires once, an hour into a seven-hour run, and the
    answer to it is a larger ``--max-new-tokens`` and a rerun of one arm. A threshold here
    would only be a number chosen to make the message stop.
    """
    payload = json.loads(record.read_text(encoding="utf-8"))
    detail = payload.get("detail") or {}
    unfinished = int(detail.get("unfinished_reasoning", 0))
    if unfinished:
        total = int(payload.get("total", 0)) or 1
        budget = (payload.get("decode") or {}).get("max_new_tokens")
        raise SystemExit(
            f"the bf16 ceiling left {unfinished}/{total} generations ({unfinished / total:.1%}) "
            f"still deliberating at {budget} new tokens, so its {payload.get('accuracy', 0):.2%} "
            f"is a decode budget and not a headroom. Raise --max-new-tokens and delete "
            f"{record}; the six quantized arms have not been spent."
        )


def check_pairable(arms: list[Arm]) -> None:
    """Every record in the panel agrees on the settings a paired test refuses to cross.

    :func:`eval_flags` makes the seven *commands* identical. This makes the seven *records*
    identical, and those stop being the same statement the moment a record was not written
    by this run -- which is what ``--resume`` is for. A leftover record's only claim to
    provenance is its filename, so one scored at a different ``--limit``, or against the
    pre-decontamination split, staples in silently and is found at compare time if at all.

    Read through the eval command's own ``_comparability``. It is private and imported
    anyway, because the alternative is a second copy of the contract living here -- and a
    field added to ``PAIRING_FIELDS`` and not to the copy would leave this guard blind to
    exactly the setting it was added to catch.

    The ceiling is the reference rather than an arbitrary member: every difference in the
    table is measured against it, so a divergence is reported as the other arm's, which is
    the direction a reader can act on.
    """
    from dynquant.commands.evaluate import _comparability

    # Every arm passed in has been scored -- the caller hands over the prefix of the panel
    # that has, not the whole panel. Filtering here instead would make the check silently
    # weaker the earlier it is called, which is the opposite of what calling it early is for.
    records = {arm.label: json.loads(Path(arm.record).read_text(encoding="utf-8")) for arm in arms}
    reference = arms[0].label
    expected = _comparability(records[reference])
    for label, record in records.items():
        found = _comparability(record)
        if found != expected:
            differed = {k: (expected[k], found[k]) for k in expected if expected[k] != found[k]}
            raise SystemExit(
                f"{label} was not scored under the same settings as {reference}: {differed} "
                f"as (reference, {label}). Their hit vectors describe different problem sets "
                f"and cannot be paired. A reused record is the usual cause."
            )


def write_manifest(
    out: Path, args: argparse.Namespace, budgets: dict[int, int], arms: list[Arm]
) -> Path:
    """The panel's manifest: the anchors, and every arm at its realised size.

    The single source of the shape ``panel_table.py`` reads. It is a function rather than a
    literal inside the run loop so a test can build its fixture through the writer instead of
    hand-copying the keys -- a fixture that agrees with a copy of the shape proves nothing
    about the shape.
    """
    manifest = out / "arms.json"
    manifest.write_text(
        json.dumps(
            {
                "model": args.model,
                "stats": args.stats,
                "moments": args.moments,
                "group_size": args.group_size,
                "tolerance": MATCH_TOLERANCE,
                "anchors": {str(width): budget for width, budget in budgets.items()},
                "arms": [
                    {
                        "label": arm.label,
                        "kind": arm.kind,
                        "anchor": arm.anchor,
                        "target_bytes": arm.target_bytes,
                        "nbytes": arm.nbytes,
                        "record": arm.record,
                        **arm.extra,
                    }
                    for arm in arms
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


def do_run(args: argparse.Namespace) -> int:
    require_one_stack()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    maps = out / "maps"
    maps.mkdir(exist_ok=True)

    budgets = anchor_bytes(args.model, args.group_size)
    arms = plan_arms(budgets)
    print(
        "anchors: " + ", ".join(f"{w}b -> {b} B ({b / 2**30:.3f} GiB)" for w, b in budgets.items()),
        flush=True,
    )

    for arm in arms:
        record = out / f"{arm.label}.json"
        save_map = maps / f"{arm.label}.json"
        # Weighed on both paths, run on one. A reused arm still has to appear in the
        # manifest at its realised size: a resume that skipped the weighing would write
        # `"nbytes": null` where the matched-byte evidence goes, and the panel would
        # read as one that never claimed a size rather than one that stopped checking.
        reused = args.resume and record.is_file()
        if reused:
            print(f"\n{arm.label}: reusing {record}", flush=True)
        if arm.kind == "ceiling":
            if not reused:
                _run(ceiling_cmd(args, arm, record), what=arm.label)
            check_uncensored(record)
        elif arm.kind == "dq":
            if not reused:
                _run(dq_inspect_cmd(args, arm, save_map), what=f"{arm.label} allocation")
            arm.nbytes = map_nbytes(save_map, str(arm.target_bytes))
            check_matched(arm)
            arm.extra["map"] = str(save_map)
            if not reused:
                _run(dq_eval_cmd(args, arm, save_map, record), what=arm.label)
        else:
            # The baselines' size is fixed by their format, so it *is* the anchor -- there
            # is nothing for `check_matched` to catch and nothing to read back.
            arm.nbytes = arm.target_bytes
            if not reused:
                _run(baseline_cmd(args, arm, record), what=arm.label)
        arm.record = str(record)
        # Against everything scored so far rather than once at the end: a reused record
        # from another run is caught after the arm that exposed it, not after six more.
        check_pairable([one for one in arms if one.record])
        # Rewritten after every arm rather than once at the end. An arm that dies takes the
        # driver with it, and a manifest written only on success would leave a seven-hour
        # panel that got as far as the fifth arm with no manifest at all -- so no table over
        # the four that did run, and nothing to read while the sixth is re-run. Arms not yet
        # scored carry `"record": null`, which the table already prints as a missing row
        # rather than as a missing comparison.
        write_manifest(out, args, budgets, arms)

    print(f"\n-> wrote {out / 'arms.json'}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    plan = sub.add_parser("plan", help="the budgets and the panel, no weights")
    plan.add_argument("--model", required=True)
    plan.add_argument("--group-size", type=int, default=128)
    plan.set_defaults(func=do_plan)

    run = sub.add_parser("run", help="every arm, in order, byte-matched")
    run.add_argument("--model", required=True, help="the fine-tuned merge")
    run.add_argument("--stats", required=True, help="the signal file the fine-tune wrote")
    run.add_argument("--moments", default=None, help="channel moments, for measured sensitivity")
    run.add_argument("--out", required=True)
    run.add_argument("--group-size", type=int, default=128)
    run.add_argument("--device", default="cuda", help="where every arm loads (default: cuda)")
    run.add_argument("--trust-remote-code", action="store_true")
    run.add_argument("--resume", action="store_true", help="skip arms whose record exists")
    run.add_argument("--calib-samples", type=int, default=256)
    run.add_argument("--seq-len", type=int, default=1024)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--split", default="test")
    run.add_argument("--shots", type=int, default=2)
    run.add_argument("--shot-seed", type=int, default=0)
    run.add_argument("--prompt-style", default="chat")
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--batch-size", type=int, default=None)
    # 1024, not the task spec's 320, and set here rather than left to default so all seven
    # arms carry the same number into their records. The base model's closure distribution
    # (report section 8) is p50 277 / p90 608 / p95 778 with 2.5% still deliberating at 1024,
    # and a query truncated mid-clause scores as a syntax error rather than as an answer.
    # That distribution does not transfer -- the arms quantize the *fine-tuned* model, which
    # was taught to answer directly -- which is why the ceiling runs first and is checked for
    # censoring before six more arms are spent under the same roof.
    run.add_argument("--max-new-tokens", type=int, default=1024)
    # Zero, because the pairing does not ride on this. `hits` is written for every item
    # whatever this is set to; `predictions` is a debugging sample of raw generations, and
    # seven arms x 400 of them is a manifest nobody reads. Raise it to look at an arm.
    run.add_argument("--keep-predictions", type=int, default=0)
    run.set_defaults(func=do_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
