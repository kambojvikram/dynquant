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
import time
from collections.abc import Sequence
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

    null_mode: str | None = None
    """Which signal-free control this arm is, or ``None`` for a real arm.

    Only ever set on ``kind == "dq"``. GPTQ and AWQ never read the signal file, so there
    is nothing in them to null; the ceiling is not allocated at all. A dq arm carrying a
    mode allocates from a permuted or flattened score under the same budget, the same
    floors and the same knapsack, and is the only thing in the panel that can say how much
    of a DynQuant margin is the *signal* rather than the shape of the map.
    """

    null_seed: int = 0
    """Ignored unless ``null_mode`` is ``"shuffle"``, which is the only stochastic one."""


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


def null_label(anchor: int, mode: str, seed: int) -> str:
    """Name a control arm, putting the seed in only when it names a different arm.

    Whether a seed distinguishes two runs of a mode is the package's question, so it is
    asked rather than restated: a stochastic mode at a second seed is a second draw and
    needs its own record, and a deterministic one at a second seed is the same arm and
    must not get one. Seed 0 keeps the bare name so the arm already banked under it
    stays the arm it was.
    """
    from dynquant.score.null import uses_seed

    return f"dq_{anchor}b_{mode[:4]}" + (f"s{seed}" if uses_seed(mode) and seed else "")


def plan_arms(
    budgets: dict[int, int],
    *,
    nulls: Sequence[str] = (),
    null_anchor: int = 3,
    null_seeds: Sequence[int] = (0,),
) -> list[Arm]:
    """The panel, in the order it should run.

    The ceiling first: it is the only arm that can fail for a reason that is not about
    quantization, and finding that out after six quantization passes wastes all six. Then
    both widths of each method rather than both methods of each width, so a method that
    cannot be built at all is discovered on its first arm.

    The controls, if any, come last and only when asked for. Two reasons, and the second is
    the load-bearing one. They are cheap relative to the panel -- an allocation and an eval,
    no calibration -- so running them at the end costs nothing that running them earlier
    would save. And the seven-arm panel is banked: ``--resume`` over a directory that
    already holds it must reuse all seven records unchanged and score only the new arms, and
    that is only true if appending a control leaves the first seven entries of this list
    exactly where they were. An arm inserted in the middle would renumber nothing -- the
    driver keys on labels, not indices -- but it would move the order the manifest is
    written in, and the manifest is what the table reads its rows off.

    Args:
        nulls: Modes from :data:`dynquant.score.null.NULL_MODES`, validated by the CLI
            against that tuple rather than against a copy here.
        null_anchor: Which budget the controls run at. Defaults to 3 because that is where
            the margin needing decomposition is: at 4 bits DynQuant is +0.78 over GPTQ and
            a control could not separate from either, while at 3 bits it is +19.13 and the
            question of what earned it is the whole reason for the arm.
        null_seeds: One control arm per seed, for the modes that read one. Several
            draws of the same permutation are what turn that rung from a point estimate
            into one carrying permutation variance, and they have to be planned
            together: one invocation per seed writes one manifest per seed, each
            describing only its own arms, and the last one written is the panel.
    """
    arms = [Arm(label="bf16", kind="ceiling", anchor=None, target_bytes=None)]
    for width in ANCHOR_WIDTHS:
        budget = budgets[width]
        for kind in ("gptq", "awq", "dq"):
            arms.append(Arm(label=f"{kind}_{width}b", kind=kind, anchor=width, target_bytes=budget))
    # Seeds outside and modes inside, with a deterministic mode planned only on the first
    # seed. That ordering is what makes a further draw an *append*: asking for seed 1 on
    # top of a directory that already banked `shuffle,uniform` at seed 0 leaves both of
    # those arms at the index the manifest already has them at. Modes outside would put
    # every extra shuffle in front of `dq_3b_unif`, moving a banked arm for a reason that
    # is not a measurement -- and the manifest order is the order the table prints in.
    from dynquant.score.null import uses_seed

    for index, seed in enumerate(null_seeds):
        for mode in nulls:
            if index and not uses_seed(mode):
                # A deterministic mode at a second seed is the same arm. Planned twice it
                # would give two arms one label, one record and one map, and the table
                # would print a replicate of a measurement that was never repeated.
                continue
            arms.append(
                Arm(
                    # Ten characters, which is the width of the table's arm column. The
                    # mode is abbreviated and the provenance is not: every artifact this
                    # arm writes carries the allocator string
                    # `sensitivity+null:shuffle(seed=0)`, the manifest carries
                    # `score_null`, and the table prints `dq-null` in the method column.
                    # A label is a filename; it is not where a control announces itself,
                    # and a wider one would misalign every row in the panel.
                    #
                    # The seed enters the label only when it distinguishes the arm, which
                    # is the package's question and not this driver's, and seed 0 keeps
                    # the bare name so the arm already banked under it stays the arm it
                    # was.
                    label=null_label(null_anchor, mode, seed),
                    kind="dq",
                    anchor=null_anchor,
                    target_bytes=budgets[null_anchor],
                    null_mode=mode,
                    null_seed=seed,
                )
            )
    return arms


def check_null_modes(spec: str) -> tuple[str, ...]:
    """Parse ``--score-null`` against the package's own list of modes.

    Read out of ``dynquant.score.null`` rather than restated as ``choices=`` on the parser,
    which would be a second copy of a registry: a mode added there and not here would be
    unreachable from the driver, and one removed there would be accepted here and fail an
    hour in. The import is lazy for the same reason every other ``dynquant`` import in this
    file is -- ``plan`` runs on machines where only the argument parsing has to work.
    """
    from dynquant.score.null import NULL_MODES

    wanted = tuple(part.strip() for part in spec.split(",") if part.strip())
    if unknown := sorted(set(wanted) - set(NULL_MODES)):
        raise SystemExit(
            f"--score-null names {unknown}; this package has {list(NULL_MODES)}. "
            f"`shuffle` permutes the driving quantity within role and `uniform` removes "
            f"it, and they bracket the question from opposite sides -- running one is not "
            f"running the other."
        )
    if len(set(wanted)) != len(wanted):
        raise SystemExit(
            f"--score-null repeats a mode: {spec!r}. Two arms would share a label and the "
            f"second would overwrite the first's record."
        )
    return wanted


def check_null_seeds(spec: str) -> tuple[int, ...]:
    """Parse ``--null-seed`` into the draws to run, refusing a repeat.

    A list rather than a scalar because permutation variance is a property of the panel and
    not of a run. Four draws asked for in four invocations write four manifests, each
    describing the arms of its own invocation and none of them describing the other three,
    and the last one written is what the table reads: the directory ends up holding four
    records and the panel naming one. Asking for them together is the only way the panel
    that answers the question is also the panel that gets banked.

    Repeats are refused for the reason a repeated mode is: two arms would share a label, and
    the second would overwrite the first's record and map while the manifest listed both --
    one measurement printed as a replication, which is the exact failure the seeded labels
    were added to prevent.
    """
    try:
        wanted = tuple(int(part) for part in spec.split(",") if part.strip())
    except ValueError as bad:
        raise SystemExit(f"--null-seed takes integers separated by commas, not {spec!r}") from bad
    if not wanted:
        raise SystemExit(
            "--null-seed is empty. It takes at least one seed, and the default is 0 -- "
            "which is the draw every banked control in this campaign was run at."
        )
    if len(set(wanted)) != len(wanted):
        raise SystemExit(
            f"--null-seed repeats a seed: {spec!r}. Two draws would share a label and the "
            f"second would overwrite the first's record."
        )
    return wanted


def check_no_arm_is_dropped(out: Path, arms: Sequence[Arm]) -> None:
    """Refuse a run that would write a manifest naming fewer arms than the directory holds.

    ``arms.json`` is not a log of the directory; it is the panel, and ``panel_table`` reads
    its rows off it rather than off a listing. So an invocation planning a different set of
    controls rewrites it, and every arm it does not plan stops being part of the panel while
    its record sits beside the new manifest saying nothing. That is how a nine-arm panel
    became an eight-arm one naming an arm that had just failed: three invocations, one seed
    each, and the last one wrote the manifest.

    Charged only against arms whose record still exists, so deleting a record is still how
    an arm is retired -- one command, and this then agrees the panel is smaller. It runs
    whether or not the run is resuming, because a run that is not resuming rewrites the
    manifest too and drops exactly the same arms.
    """
    manifest = out / "arms.json"
    if not manifest.is_file():
        return
    planned = {arm.label for arm in arms}
    previous = json.loads(manifest.read_text(encoding="utf-8"))
    dropped = [
        entry["label"]
        for entry in previous.get("arms", [])
        if entry["label"] not in planned and (out / (entry["label"] + ".json")).is_file()
    ]
    if dropped:
        raise SystemExit(
            f"{manifest} names {dropped}, whose records are in {out}, and this run does not "
            f"plan them. Writing the manifest would leave the records on disk and take the "
            f"arms out of the panel, which is a smaller panel and not a measurement. Ask "
            f"for them -- --score-null and --null-seed both take lists -- or delete their "
            f"records to retire them."
        )


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

    ``--experts-impl`` is here for a stronger reason than the rest. The others are settings
    the arms could in principle disagree about; this one is which *arithmetic* the model
    runs, and the arms disagreed about it silently. A dq arm encodes in place and keeps
    ``post_init``'s ``grouped_mm``; a GPTQ or AWQ arm arrives from ``llm-compressor`` with
    its banks linearised into per-expert ``Linear`` modules and computes eager; on
    LFM2.5-8B-A1B those differ on 1.24% of teacher-forced tokens, 0.29x the quantization
    effect and the same order as the margin between the two.
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
        # Stated rather than inherited, exactly like the decode budget above: the default
        # lives in `dynquant eval` and this driver is the thing that has to be able to say
        # what every arm ran, including after that default moves.
        "--experts-impl",
        args.experts_impl,
        # Pinned, and the reason is the same shape as `--experts-impl` above: the eval
        # default is *derived* -- every registry source whose contexts carry rows -- so
        # adding a source widens the scored set without touching this file. Spider did
        # exactly that. An unpinned panel would have scored a different mixture than the
        # arms already banked and the pairing guard would have caught it as a mismatch
        # rather than as the silent denominator change it actually is.
        "--sources",
        *args.sources,
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
    # Last, and after `--moments`, because that is the order the allocator applies them in:
    # the null runs on whatever the scoring and the sensitivity pricing produced, so a
    # control on a panel that priced from measured `dL` nulls the measured table too. A
    # null applied before the pricing would leave 8.5% of this checkpoint's parameters
    # still priced by their own moments and the arm would be a partial control reported as
    # a whole one. The flags only appear on an arm that asked for them, so the six real
    # arms' commands are byte-identical to the ones the banked panel ran.
    if arm.null_mode:
        cmd += ["--score-null", arm.null_mode, "--null-seed", str(arm.null_seed)]
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
    """Run one arm's subprocess, stamped, and say how long it took.

    These two lines are the only record of where a panel's hours went. An eval record
    carries its own ``seconds``, so the scoring half is self-timing; the quantization
    half is not, and on a baseline arm that half is a 256-sample calibration pass over
    8 B parameters whose cost this campaign has never measured. Without a stamp either
    side of the subprocess the next panel guesses the same number again, and the guess
    decides how many items it can afford.

    The elapsed line is printed before the return code is checked, so the arm that
    *failed* is the one whose duration is recorded -- which is when it is worth most.
    """
    started = time.time()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
    print(f"\n[{stamp}] $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    print(f"[{what}] exit {result.returncode} after {time.time() - started:.1f}s", flush=True)
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
    arms = plan_arms(
        budgets,
        nulls=check_null_modes(args.score_null),
        null_anchor=args.null_anchor,
        null_seeds=check_null_seeds(args.null_seed),
    )
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
                        **(
                            {"score_null": {"mode": arm.null_mode, "seed": arm.null_seed}}
                            if arm.null_mode
                            else {}
                        ),
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

    One field is reported and not refused, and the reason is in the comment on
    ``computation`` below. It is the only member of the contract that describes how an
    answer was computed rather than which question was asked.
    """
    from dynquant.commands.evaluate import _comparability, problem_set_difference

    # Every arm passed in has been scored -- the caller hands over the prefix of the panel
    # that has, not the whole panel. Filtering here instead would make the check silently
    # weaker the earlier it is called, which is the opposite of what calling it early is for.
    records = {arm.label: json.loads(Path(arm.record).read_text(encoding="utf-8")) for arm in arms}
    reference = arms[0].label
    # Held out of the refusal, and out of it by name rather than by value, so a field added
    # to `EXPERTS_PAIRING_FIELDS` is held out with it instead of turning fatal unannounced.
    #
    # Every other member of the contract names a question that was asked -- the task, the
    # split, the shot seed, the decode budget. Two records that disagree on one of those
    # were scored over different items, so their hit vectors do not line up element-wise
    # and a paired test across them is arithmetic on unrelated vectors. `experts.ran` is
    # not that. The items are the same items in the same order and the vectors pair; what a
    # difference costs is the *reading* of the delta, and the reading is priced per
    # comparison by `panel_table.print_comparisons`, against the 1.24% of teacher-forced
    # tokens the two dispatches disagree on.
    #
    # So refusing it here would make a panel un-finishable by a fact the table renders.
    # `--rescore bf16,dq_4b,dq_3b --experts-impl eager` is exactly that shape: three arms
    # score again under a dispatch the four reused baselines predate, the disagreement is
    # explained in full by the command that was typed, and a fatal check would stop the
    # driver on the second arm -- after the ceiling re-score and before any DynQuant arm.
    expected = _comparability(records[reference])
    reported = []
    for label, record in records.items():
        problem = problem_set_difference(records[reference], record)
        if problem:
            raise SystemExit(
                f"{label} was not scored under the same settings as {reference}: {problem} "
                f"as (reference, {label}). Their hit vectors describe different problem sets "
                f"and cannot be paired. A reused record is the usual cause."
            )
        if _comparability(record) != expected:
            reported.append(label)
    if reported:
        # One line per arm boundary rather than one per mismatching arm, because this runs
        # again over the growing prefix after every arm and the same four would otherwise
        # accumulate down the log.
        print(
            f"  note: {len(reported)} arm(s) disagree with {reference} on the expert "
            f"dispatch and pair anyway: {', '.join(reported)}. Same items, same order, "
            f"different arithmetic -- panel_table marks the comparisons it reaches.",
            flush=True,
        )


def rescored_labels(spec: str, arms: list[Arm]) -> frozenset[str]:
    """Arms to score again from the artifacts they already hold.

    `--resume` answers "which arms are done"; this answers a different question, and the
    difference is the whole point. An arm that has to be scored again because the *scorer*
    changed has not had its allocation invalidated, and re-deriving the map is not a free
    no-op that happens to cost a few minutes: it is a second change, made silently, in the
    same run as the one being measured. The eager re-score of section 8 is exactly this
    shape -- the dispatch moved, the bit map did not -- and a re-run that re-allocated
    would compare a new dispatch on a new map against an old dispatch on an old one.

    So a rescored DynQuant arm reuses `maps/<label>.json` and refuses if it is not there,
    rather than generating one. It is still weighed and still checked against its anchor:
    reusing an allocation is not the same as trusting it.

    Refuses an unknown label instead of ignoring it, because the failure it prevents is a
    typo that silently rescores nothing and reports success.
    """
    wanted = frozenset(part.strip() for part in spec.split(",") if part.strip())
    known = {arm.label for arm in arms}
    if unknown := sorted(wanted - known):
        raise SystemExit(
            f"--rescore names {unknown}, which this panel does not plan. It plans {sorted(known)}."
        )
    return wanted


def check_resumable(
    out: Path, args: argparse.Namespace, arms: list[Arm], *, rescore: frozenset[str]
) -> None:
    """A reused record was scored by *some* run. Check that it was scored by this one.

    :func:`check_pairable` compares records against each other through ``_comparability``,
    which is the contract a paired test needs: task, backend, split, shots, shot seed, limit.
    None of those fields name the *model*, and none of them can be older than anything. So
    two records scored on two different merges at identical settings pair cleanly, and a
    record scored before the signal file it should have read pairs cleanly with one scored
    after it. Both produce a panel that reports a difference between methods and is measuring
    a difference in the checkpoint.

    Two questions, because they fail differently. The manifest records which model and which
    signal file the previous run was handed, and a resume that changes either is a new panel
    wearing the old one's directory. Then each surviving record is checked against the mtimes
    of the inputs it claims to have been scored from -- a skip-if-output-exists guard cannot
    see that its output predates its input unless it is told to look.

    The stats file is charged only against the DynQuant arms, because the baselines never read
    it. Regenerating the signal without retraining invalidates two arms, and refusing the four
    that are still good would make the cheapest correct fix look as expensive as the panel.

    Refuses rather than repairs. Deleting the directory is one command and always right;
    deciding which of seven records survived a change of inputs is neither.
    """
    manifest = out / "arms.json"
    if manifest.is_file():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        current = {
            "model": args.model,
            "stats": args.stats,
            "moments": args.moments,
            "group_size": args.group_size,
        }
        differed = {k: (previous.get(k), v) for k, v in current.items() if previous.get(k) != v}
        if differed:
            raise SystemExit(
                f"{manifest} was written for {differed} as (previous, now). Resuming would "
                f"reuse records scored against the previous inputs and table them beside arms "
                f"scored against these. Delete {out} and run the panel whole."
            )

    stale = []
    for arm in arms:
        # The check has to follow whatever is being *kept*, which is not always the record.
        # A rescored arm's record is about to be overwritten, so its age says nothing about
        # this run; what it carries forward instead is its allocation map, and that is the
        # artifact that has to postdate the inputs. A rescored arm with no map to keep is
        # refused in `do_run` rather than here, where there is nothing yet to be stale.
        rescoring = arm.label in rescore
        if rescoring and arm.kind != "dq":
            continue
        kept = out / "maps" / f"{arm.label}.json" if rescoring else out / f"{arm.label}.json"
        if not kept.is_file():
            continue
        written = kept.stat().st_mtime
        sources = {"the model": Path(args.model) / "config.json"}
        if arm.kind == "dq":
            sources["the stats file"] = Path(args.stats)
        for what, source in sources.items():
            if source.exists() and written < source.stat().st_mtime:
                # Relative, because a map and a record share a basename and the message has
                # to say which of the two is the one that went stale.
                stale.append(f"{kept.relative_to(out)} predates {what}, {source}")
    if stale:
        raise SystemExit(
            "\n".join(
                ["these records were written before the inputs they should have scored:"]
                + [f"  {line}" for line in stale]
                + [f"Delete them, or delete {out} and run the panel whole."]
            )
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
    arms = plan_arms(
        budgets,
        nulls=check_null_modes(args.score_null),
        null_anchor=args.null_anchor,
        null_seeds=check_null_seeds(args.null_seed),
    )
    check_no_arm_is_dropped(out, arms)
    rescore = rescored_labels(args.rescore, arms)
    # `--rescore` implies the resume checks and the resume skipping. Naming three arms to
    # score again and having the other four score again too is not a thing anyone means,
    # and the input checks are wanted *more* here, not less: a re-score reuses more of the
    # previous run than a resume does.
    resuming = args.resume or bool(rescore)
    if resuming:
        check_resumable(out, args, arms, rescore=rescore)
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
        rescoring = arm.label in rescore
        reused = resuming and record.is_file() and not rescoring
        if reused:
            print(f"\n{arm.label}: reusing {record}", flush=True)
        if rescoring:
            print(f"\n{arm.label}: scoring again, allocation unchanged", flush=True)
        if arm.kind == "ceiling":
            if not reused:
                _run(ceiling_cmd(args, arm, record), what=arm.label)
            check_uncensored(record)
        elif arm.kind == "dq":
            if rescoring and not save_map.is_file():
                raise SystemExit(
                    f"--rescore {arm.label} reuses the allocation this arm was already "
                    f"given, and {save_map} is not there. Drop it from --rescore to "
                    f"allocate it afresh; a re-score that quietly re-allocates changes two "
                    f"things and reports one."
                )
            if not reused and not rescoring:
                _run(dq_inspect_cmd(args, arm, save_map), what=f"{arm.label} allocation")
            # On both paths, including the rescored one. Reusing a map is not trusting it:
            # the arm still has to appear in the manifest at its realised size and still
            # has to hit the anchor, or the re-score is a matched-byte claim nobody checked.
            arm.nbytes = map_nbytes(save_map, str(arm.target_bytes))
            check_matched(arm)
            arm.extra["map"] = str(save_map)
            if arm.null_mode:
                # In the manifest as well as in the map, because the table reads the
                # manifest and an arm that is a control has to be one in the row a reader
                # sees, not only in a file they would have to go and open.
                arm.extra["score_null"] = {"mode": arm.null_mode, "seed": arm.null_seed}
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


def add_null_flags(parser: argparse.ArgumentParser) -> None:
    """``--score-null`` on both stages, so ``plan`` shows the panel ``run`` would run.

    Opt-in and empty by default. The controls answer a question the seven-arm panel does
    not ask, and a driver that ran them unasked would change the shape of every banked
    manifest and every table built from one.
    """
    parser.add_argument(
        "--score-null",
        default="",
        metavar="MODE[,MODE...]",
        help=(
            "append a signal-free control arm per mode: same graph, roles, floors, budget, "
            "byte accounting, knapsack and quantizer, with only the correspondence between "
            "a module and what the fine-tune said about it removed. `shuffle` permutes the "
            "driving quantity within role; `uniform` scores every module 1.0 and consults "
            "no sensitivity table. What the real arm has over these is the signal's share"
        ),
    )
    parser.add_argument(
        "--null-anchor",
        type=int,
        default=3,
        choices=ANCHOR_WIDTHS,
        help="which budget the controls run at (default: 3, where the margin is)",
    )
    parser.add_argument(
        "--null-seed",
        default="0",
        metavar="SEED[,SEED...]",
        help=(
            "seeds for the modes that read one, one control arm per seed (default: 0). "
            "`uniform` ignores them and is planned once however many are given. More than "
            "one draw is what turns the shuffle rung from a point estimate into one "
            "carrying permutation variance"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    plan = sub.add_parser("plan", help="the budgets and the panel, no weights")
    plan.add_argument("--model", required=True)
    plan.add_argument("--group-size", type=int, default=128)
    add_null_flags(plan)
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
    run.add_argument(
        "--rescore",
        default="",
        metavar="LABEL[,LABEL...]",
        help=(
            "score these arms again from the allocation they already have, and resume the "
            "rest. For when the scorer changed and the bit map did not -- a plain rerun "
            "would re-allocate, which is a second change in the run that is measuring the "
            "first. Refuses an arm whose map is missing rather than making one"
        ),
    )
    run.add_argument("--calib-samples", type=int, default=256)
    run.add_argument("--seq-len", type=int, default=1024)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--split", default="test")
    run.add_argument("--shots", type=int, default=2)
    run.add_argument("--shot-seed", type=int, default=0)
    run.add_argument("--prompt-style", default="chat")
    run.add_argument(
        "--sources",
        nargs="+",
        default=["gretel", "wikisql"],
        metavar="NAME",
        help=(
            "text2sql sources to score. Defaults to the two this panel's banked arms "
            "were scored on, not to the eval default, which grows with the registry"
        ),
    )
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
    # `eager` on every arm, because it is the only dispatch all seven can share: a
    # linearised baseline has no `*Experts` module left to put on `grouped_mm`, and a
    # packed download is forced to `eager` whatever the panel did. `auto` is for measuring
    # the dispatch itself, which is not what a panel does.
    run.add_argument("--experts-impl", default="eager", choices=("eager", "auto"))
    add_null_flags(run)
    run.set_defaults(func=do_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
