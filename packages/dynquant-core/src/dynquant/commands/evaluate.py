"""``dynquant eval`` -- score a model on a task, and compare two scores honestly.

The number this prints is only worth anything if the *only* thing that differed
between two runs was the weights. That is a property of the harness, not of the
caller's discipline, so everything that could differ is fixed here and recorded in
the output: greedy decode, one prompt format, a fixed few-shot prefix chosen by
seed from the train split, and the same scorer whatever produced the text. The
few-shot prefix is kept for a fine-tuned model too -- dropping it would confound
"got better at the task" with "learned this output format".

Two things this deliberately does that a simpler evaluator would not:

**It writes the per-problem correctness vector.** ``--out`` records one boolean per
problem, because the comparison that matters -- allocated against uniform at the
same budget -- is *paired*: the same problems, in the same order, scored twice. An
unpaired test on this data is roughly twice as wide, and the first version of this
harness stored only the counts, which meant recovering the vector cost a re-run of
every arm on the GPU.

**It reports the chance floor.** A 5-way multiple choice bottoms out at 20%, so a
destroyed model returns to 20% rather than to zero, and a table without the floor
makes a collapsed arm look like a mildly damaged one.

``--map`` takes the *unquantized* model plus a bit map and applies the widths in
memory, so no arm needs a checkpoint written for it. ``--map-apply pack`` (the
default) swaps the named modules onto the packed runtime, so the weights stay
quantized in VRAM and the memory figure is real. ``--map-apply encode`` writes the
same encoder's reconstruction back in the compute dtype: the same values, fp16
size, and the only mode that reaches a weight held as a bare parameter rather than
a module -- which on a batched-expert MoE is most of the model. A directory written
by ``dynquant quantize`` is the same thing as ``encode``, spelled on disk.

**It pins the experts dispatch.** The two modes hold the same values and do not, on
a MoE, run the same computation: packing moves the model off the default dispatch,
because a packed bank has to be indexed one expert at a time, while encoding leaves
whatever the model chose at ``post_init``. On LFM2.5-8B-A1B ``eager`` and that default
disagree on 1.24% of teacher-forced tokens, 0.29x what quantizing to 4 bits does --
the same order as the margins this command exists to measure. So every run is pinned
to one dispatch and the record says which one ran. See ``--experts-impl``.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import _shared

if TYPE_CHECKING:
    import argparse

__all__ = [
    "DECODE_PAIRING_FIELDS",
    "DETAIL_PAIRING_FIELDS",
    "EXPERTS_PAIRING_FIELDS",
    "PAIRING_FIELDS",
    "TASKS",
    "run",
]

#: Everything two records must agree on before their hit vectors may be paired.
#:
#: ``backend`` is here even though G4 measures the two runtimes as equivalent in
#: *score*. Equal totals are not identical per-item outcomes, and McNemar reads exactly
#: the items the two arms disagree on -- so a cross-backend pair would put engine
#: disagreement into the one cell the test counts.
PAIRING_FIELDS = ("task", "backend", "split", "shots", "shot_seed", "limit")

#: The same contract, for settings the record keeps under ``decode``.
#:
#: The budget is here because of what it did once. On text2sql, a 256-token cap on a
#: model that deliberates before answering scored 5.50%, and that number was taken for a
#: headroom measurement and a campaign was nearly configured off it; at 1024 the same
#: model on the same 400 problems scores 57.75%. Two arms at different caps are not
#: being asked the same question, and until now nothing in the guard said so.
#:
#: ``batch_size`` is deliberately *not* here. Left-padded batched decode can perturb the
#: last bits of a logit, so two batch sizes are not guaranteed identical per-item
#: outcomes -- but the prompts and the problems are the same, which is what pairing is
#: about, and requiring a match would stop a 3-bit arm from running at the batch size its
#: memory allows. The residual risk is real and is accepted here rather than unnoticed.
DECODE_PAIRING_FIELDS = ("max_new_tokens",)

#: The same contract again, for settings only the *task* knows.
#:
#: ``prompt_style`` is resolved from the tokenizer, not passed on the command line, so
#: it is the one pairing field two arms can disagree about while their commands are
#: byte-identical. A quantized checkpoint whose saved tokenizer lost its chat template
#: resolves to ``completion`` where the ceiling resolves to ``chat``; both arms then run
#: clean, and the difference between being asked a chat question and a completion
#: question is reported as the effect of quantization.
#:
#: Read out of ``detail`` because that is where a task's own metrics already go. Tasks
#: that record no style are ``_ABSENT`` on both sides and pair as before -- absence is
#: only a mismatch against a record that has one.
DETAIL_PAIRING_FIELDS = ("prompt_style",)

#: The same contract once more, for the *computation* rather than for a setting.
#:
#: Three arms of one panel reach the scorer on three different experts dispatches
#: without anyone choosing it -- packed arms forced to ``eager``, encoded arms left on
#: ``post_init``'s ``grouped_mm``, and a baseline whose banks ``llm-compressor`` has
#: already rewritten into per-expert ``Linear`` modules, on which the setting is inert
#: because there is no batched bank for it to act on. Read out of the ``experts`` block,
#: where the run writes what it found and what it ran.
#:
#: Optional, like ``prompt_style``, and for a reason worth stating rather than inheriting.
#: A dense model has no such dispatch and never will, so absence has to pair with absence
#: -- which means a record written before this field existed also pairs with a dense one,
#: and with another record written before this field existed. What the field buys is that
#: no run from here on is unable to say.
#:
#: An older record is not always beyond recovery, and the guard is deliberately not the
#: place that recovers it. A ``baselines_lfm2`` arm has a ``.quant.json`` beside it
#: reporting ``banks_after: 0``, counted in the process that then scored that same object,
#: and a model with no batched bank has no grouped kernel to take. That settles what the
#: arithmetic was; it does not put the answer in this record, and pairing reads records.
#: ``panel_table.print_dispatch`` is where the two are told apart, because a reader needs
#: to know which ``NOT PAIRED`` lines are about computation and which are about
#: bookkeeping.
EXPERTS_PAIRING_FIELDS = ("ran",)

#: Comparability keys whose absence from *this* run's record is legitimate.
#:
#: Every other pairing field is written by ``dynquant eval`` on every run, so missing one
#: is a bug in the command and is raised as such. The detail block is different: it is the
#: task's, three tasks carry none at all, and the ones that do carry different keys. So
#: absence here has to mean "this task does not report a style", which pairs cleanly
#: against another record that does not report one and refuses against one that does.
_OPTIONAL_COMPARABILITY = frozenset(
    [f"detail.{field}" for field in DETAIL_PAIRING_FIELDS]
    + [f"experts.{field}" for field in EXPERTS_PAIRING_FIELDS]
)

#: Distinguishes "this record does not carry the field" from "it carries ``None``".
#: ``split`` is legitimately ``None`` for a single-set dataset, so absence cannot be
#: spelled that way.
_ABSENT = object()


class _TaskSpec:
    """Per-task defaults, chosen once so no two runs can pick different ones.

    Every field is something two runs could otherwise disagree about. The ones the
    phase-3 tasks added are less obvious than a decode budget and are worth naming:

    ``split`` and ``shot_split`` are ``None`` for a task whose dataset has a single
    set and whose loader therefore takes no split argument (HumanEval), and for a task
    that takes no few-shot prefix at all (IFEval, HumanEval). A default of ``"test"``
    imposed by the argument parser cannot express either -- it would ask a train-only
    dataset for a test split, and hand ``shots=`` to a function that has no such
    parameter.

    ``unscored`` names the field counting generations that produced nothing to score.
    Every task has one, and every one of them means the same thing -- the model
    stopped producing the format rather than got the answer wrong -- but they are
    spelled differently per task (``unparseable`` where a number is expected,
    ``empty`` where any text would do). Declared here rather than sniffed off the
    result, so a renamed field is an import-time error and not a silent zero.
    """

    def __init__(
        self,
        key: str,
        *,
        shots: int,
        chance: float,
        max_new_tokens: int,
        max_prompt_tokens: int,
        batch_size: int,
        split: str | None = "test",
        shot_split: str | None = "train",
        add_special_tokens: bool = True,
        unscored: str = "unparseable",
        executes_code: bool = False,
        takes_style: bool = False,
        unverifiable: bool = False,
        detail: bool = False,
    ) -> None:
        self.key = key
        self.shots = shots
        self.chance = chance
        self.max_new_tokens = max_new_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.batch_size = batch_size
        self.split = split
        self.shot_split = shot_split
        self.add_special_tokens = add_special_tokens
        self.unscored = unscored
        self.executes_code = executes_code
        self.takes_style = takes_style
        self.unverifiable = unverifiable
        self.detail = detail

    @property
    def takes_shots(self) -> bool:
        return self.shot_split is not None

    def _module(self) -> Any:
        """Import the task module on demand.

        Deferred because each one imports ``datasets`` transitively, and building
        the argument parser must not pay for three dataset backends to offer
        ``--task`` as a choice.
        """
        from importlib import import_module

        return import_module(f"dynquant.eval.{self.key}")

    def load(self, split: str | None) -> list[Any]:
        loader = getattr(self._module(), f"load_{self.key}")
        return loader() if split is None else loader(split)  # type: ignore[no-any-return]

    def unscored_count(self, result: Any) -> int:
        return int(getattr(result, self.unscored))

    def detail_of(self, result: Any) -> dict[str, Any] | None:
        """The task's own metrics, for tasks that report more than one number.

        IFEval has four official metrics, and the code tasks distinguish a timeout from
        a wrong answer. Dropping those would leave the record unable to say which of
        the two a regression was. The three single-number tasks are already recorded
        in full by the fields around this one, so they carry no detail block rather
        than a duplicate one.
        """
        return result.as_dict() if self.detail else None

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        return getattr(self._module(), f"evaluate_{self.key}")(*args, **kwargs)


TASKS = {
    # Five shots, a full worked solution to generate, and a 0% floor.
    "gsm8k": _TaskSpec(
        "gsm8k",
        shots=5,
        chance=0.0,
        max_new_tokens=320,
        max_prompt_tokens=2048,
        batch_size=48,
    ),
    # Two shots, not five: a CaseHOLD prompt is ~400 tokens against GSM8K's ~90, so
    # five would push the prefix past 2k for no measured benefit. The answer is one
    # digit, so eight new tokens is room for " 3" plus punctuation.
    "casehold": _TaskSpec(
        "casehold",
        shots=2,
        chance=0.2,
        max_new_tokens=8,
        max_prompt_tokens=3072,
        batch_size=32,
    ),
    # Four shots. The prompt is dominated by the 77-line intent header (~615 of
    # ~710 tokens) rather than by the exemplars, which are one sentence each, so
    # shots are nearly free here in a way they are not on CaseHOLD. Six new tokens
    # is room for a two-digit index plus punctuation. 1536 leaves headroom over the
    # header so nothing can truncate it -- front-truncation would cut the list the
    # answer indexes into and score the model on a numbering it never saw.
    "banking77": _TaskSpec(
        "banking77",
        shots=4,
        chance=1.0 / 77,
        max_new_tokens=6,
        max_prompt_tokens=1536,
        batch_size=32,
    ),
    # IFEval ships a single split, named `train`, and takes no few-shot prefix: the
    # constraint *is* the instruction, and an exemplar would demonstrate obeying a
    # different constraint. 1024 new tokens because the constraints are frequently
    # length constraints -- "at least 400 words", "write 5 paragraphs" -- and a cap
    # that truncates the answer scores the cap rather than the model.
    # `add_special_tokens=False` because the prompt is a chat template that already
    # carries its own BOS; leaving it on gives Llama-3 and Gemma-3 two, which is no
    # error and a few points of damage.
    "ifeval": _TaskSpec(
        "ifeval",
        shots=0,
        chance=0.0,
        max_new_tokens=1024,
        max_prompt_tokens=2048,
        batch_size=16,
        split="train",
        shot_split=None,
        add_special_tokens=False,
        unscored="empty",
        unverifiable=True,
        detail=True,
    ),
    # HumanEval is 164 problems in one set, so `load_humaneval` takes no split
    # argument at all. No shots: the docstring in the prompt is the specification.
    "humaneval": _TaskSpec(
        "humaneval",
        shots=0,
        chance=0.0,
        max_new_tokens=1024,
        max_prompt_tokens=2048,
        batch_size=16,
        split=None,
        shot_split=None,
        unscored="empty",
        executes_code=True,
        takes_style=True,
        detail=True,
    ),
    # Text-to-SQL, scored by running the query. Three datasets mixed and balanced by
    # `load_text2sql`, so the headline is not an average weighted by whichever source
    # survives its admission filter most often; `detail` carries the per-source split.
    #
    # Two shots, from a `shots` pseudo-split rather than from `train`: the training
    # mixture is ~230k rows and every one of them costs a gold-query execution at load
    # time, which is hours to choose two exemplars. It resolves to a bounded slice of
    # `train`, disjoint from `test` in both scored sources.
    #
    # 320 new tokens because a Gretel gold with a CTE and two joins runs long, and a
    # query truncated mid-clause is scored as a syntax error rather than as an answer.
    # `unscored="unparseable"` is generations with no SQL in them at all, which on a
    # zero-floor task is the first thing to move when quantization breaks format
    # compliance rather than accuracy.
    "text2sql": _TaskSpec(
        "text2sql",
        shots=2,
        chance=0.0,
        max_new_tokens=320,
        max_prompt_tokens=3072,
        batch_size=32,
        split="test",
        shot_split="shots",
        takes_style=True,
        detail=True,
    ),
    # MBPP's exemplars are its own `prompt` split, which is not scored. The budgets
    # here are the chat-framing ones, which are the larger of the two the task defines;
    # under the completion framing they only mean the generation ends at the task's own
    # stop sequence rather than at the cap, so no solution is truncated either way.
    "mbpp": _TaskSpec(
        "mbpp",
        shots=3,
        chance=0.0,
        max_new_tokens=1024,
        max_prompt_tokens=2048,
        batch_size=16,
        split="test",
        shot_split="prompt",
        unscored="empty",
        executes_code=True,
        takes_style=True,
        detail=True,
    ),
}


def run(args: argparse.Namespace, *, model: Any = None) -> int:
    """Score a model on a task and write the record.

    ``model`` exists for drivers that build a runtime the CLI cannot name. A
    post-training baseline is the case: ``llm-compressor`` returns a quantized model
    object, and at 3 bits ``compressed-tensors`` has no packed format to save it in, so
    the checkpoint a path would point at is a dequantized bf16 copy of the same weights
    at four times the size. Passing the object scores the arm the CLI would have scored
    without writing 17 GB first.

    It is a parameter here rather than a second evaluator beside this one because
    everything below this line is what makes two numbers comparable -- the prompt, the
    shot prefix and its seed, the decode settings, the scorer, the per-item hit vector,
    and ``_pairing``, which is what a later McNemar test checks before it agrees to pair
    two records. ``experiments/four_point`` has its own ``run_eval``, and the cost of
    that is exactly this: its records cannot be compared against a ``dynquant eval``
    record without arguing that two implementations of the same settings agree.

    A caller that passes ``model`` still owns ``args.model``: it is the record's
    provenance field and the default label, so it has to describe where those weights
    came from. Nothing here can check that, which is why it is said out loud. It is also
    the default for ``--tokenizer``, and that one *is* resolved as a path -- so a caller
    that qualifies the field to say which recipe produced the weights has to pass
    ``--tokenizer`` too, or the run dies loading a tokenizer from a string that was
    never meant to be one.
    """
    from dynquant._version import __version__
    from dynquant.errors import DynQuantError
    from dynquant.eval.harness import EvalConfig

    spec = TASKS[args.task]
    split, shot_split, n_shots = _resolve_splits(spec, args)

    examples = spec.load(split)
    shots = _pick_shots(spec, n_shots, seed=args.shot_seed, split=shot_split)
    source = f"the {split} split" if split is not None else "its only split"
    prefix = f"{len(shots)} shot(s) from {shot_split} at seed {args.shot_seed}"
    print(
        f"{args.task}: {len(examples)} examples from {source}, "
        f"{prefix if spec.takes_shots else 'no few-shot prefix'}",
        flush=True,
    )

    # Built before the runtime, not after: the vLLM engine sizes its context from these
    # numbers, and an engine shorter than the harness's own limits truncates prompts the
    # transformers path would have scored in full.
    config = EvalConfig(
        max_new_tokens=args.max_new_tokens or spec.max_new_tokens,
        batch_size=args.batch_size or spec.batch_size,
        max_prompt_tokens=args.max_prompt_tokens or spec.max_prompt_tokens,
        add_special_tokens=spec.add_special_tokens,
        limit=args.limit,
    )

    if model is None:
        model, packed = _load_runtime(args, config)
    elif args.map is not None:
        # --map builds the packed runtime *from* an unquantized model, so honouring both
        # would mean quantizing a model that is already quantized. Refused rather than
        # resolved: either answer is a guess about which of the two the caller meant, and
        # the wrong guess produces a plausible number under the other one's label.
        raise DynQuantError(
            "an in-memory model was passed and --map was also set. --map quantizes the "
            "model it is given; a model that arrives already quantized cannot also be "
            "the input to that. Pass one or the other."
        )
    else:
        # No packing record, because nothing here packed anything. A caller with its own
        # size accounting puts it in its own record beside this one -- writing it into
        # `packed` would claim this command measured it.
        packed = None

    # After the branch and not inside `_load_runtime`, because a baseline arrives through
    # `model=` and a ceiling applies no map at all -- so this is the only point in the
    # command that every arm of a panel passes through.
    experts = _pin_experts_dispatch(model, args)

    tokenizer = _shared.load_tokenizer(
        args.tokenizer or args.model, trust_remote_code=args.trust_remote_code
    )
    label = args.label or f"{args.task}:{Path(args.model).name}"

    started = time.time()
    result = spec.evaluate(
        model,
        tokenizer,
        examples,
        label=label,
        config=config,
        progress=None if args.quiet else _shared.progress_printer(args.task, every=200),
        keep_predictions=args.keep_predictions,
        **_task_kwargs(spec, args, shots),
    )
    elapsed = time.time() - started
    print("\n" + result.summary(), flush=True)

    record: dict[str, Any] = {
        "dynquant_core": __version__,
        "label": result.label,
        "model": args.model,
        **_pairing(args, split=split, n_shots=n_shots),
        "accuracy": result.accuracy,
        # Counted from the vector rather than read off a per-task field, so the
        # headline number and the paired test can never disagree about which problems
        # were right -- and so one definition covers "exact match", "prompt-level
        # strict" and "pass@1" without the command knowing which it is looking at.
        "correct": sum(1 for hit in result.hits if hit),
        "total": result.total,
        "unparseable": spec.unscored_count(result),
        "detail": spec.detail_of(result),
        "chance": spec.chance,
        "seconds": round(elapsed, 1),
        "decode": {
            "max_new_tokens": config.max_new_tokens,
            "batch_size": config.batch_size,
            "max_prompt_tokens": config.max_prompt_tokens,
            "greedy": True,
        },
        "packed": packed,
        "experts": experts,
        "hits": result.hits,
        "predictions": result.predictions,
    }

    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        print(f"-> wrote {destination}", flush=True)

    comparison: dict[str, Any] | None = None
    if args.compare:
        comparison = _compare(record, args.compare)

    if args.json:
        payload = dict(record)
        if comparison is not None:
            payload["comparison"] = comparison
        print(json.dumps(payload, indent=2, default=str))

    if args.min_accuracy is not None and result.accuracy < args.min_accuracy:
        raise DynQuantError(
            f"accuracy {result.accuracy:.4f} is below --min-accuracy {args.min_accuracy:.4f}"
        )
    return 0


def _pairing(args: argparse.Namespace, *, split: str | None, n_shots: int) -> dict[str, Any]:
    """The settings two runs must share before their hit vectors may be paired.

    ``PAIRING_FIELDS`` names them; this builds them. Keeping the two beside each other
    is the point: written inline in the record, a field could be added to the contract
    and never recorded -- which makes every comparison against a record written by the
    same code raise -- or recorded and dropped from the contract, which makes the
    comparison guard blind to exactly the setting it was added to catch. Neither
    direction is visible to a test that walks ``PAIRING_FIELDS``, because a shrunken
    tuple just yields fewer cases and stays green.
    """
    from dynquant.errors import DynQuantError

    values: dict[str, Any] = {
        "task": args.task,
        "backend": args.backend,
        "split": split,
        "shots": n_shots,
        "shot_seed": args.shot_seed,
        "limit": args.limit,
    }
    if set(values) != set(PAIRING_FIELDS):
        raise DynQuantError(
            f"the record pairs on {sorted(values)} but PAIRING_FIELDS names "
            f"{sorted(PAIRING_FIELDS)}. That is a bug in `dynquant eval`: the two have "
            f"to describe the same settings or the comparison guard stops guarding."
        )
    return values


def _load_runtime(args: argparse.Namespace, config: Any) -> tuple[Any, dict[str, Any] | None]:
    """Whatever the harness will generate with, and what it cost to pack.

    The harness only ever calls ``generate_ids``, so a vLLM engine substitutes for a
    ``transformers`` model without the scorers knowing which they were handed. That is
    the property ``scripts/gate_runtime_parity.py`` measures on real hardware; it is not
    assumed here.

    Serving the campaign through vLLM is the difference between a week and a month at
    S4's volume, which is why this lives in the command rather than in a script beside
    it -- a second evaluator built to reach the engine would be a second place for the
    prompt to be assembled differently.
    """
    from dynquant.errors import DynQuantError

    if args.backend == "transformers":
        model = _shared.load_model(
            args.model,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        model.config.use_cache = True
        return model, (_apply_map(model, args) if args.map is not None else None)

    if args.map is not None:
        raise DynQuantError(
            "--map swaps modules on a `transformers` model that is already in memory, "
            "and the vLLM backend never builds one. Write the checkpoint first with "
            "`dynquant quantize --map ...` and point --model at that directory, which "
            "is the configuration a server actually loads."
        )

    from dynquant.eval.backends import VllmBackend

    engine_kwargs: dict[str, Any] = {
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len or (config.max_prompt_tokens + config.max_new_tokens),
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.enforce_eager,
    }
    if args.quantization:
        engine_kwargs["quantization"] = args.quantization
    if args.tensor_parallel_size > 1:
        engine_kwargs["tensor_parallel_size"] = args.tensor_parallel_size

    print(f"vllm: building engine with {engine_kwargs}", flush=True)
    return VllmBackend.from_pretrained(args.model, **engine_kwargs), None


def _resolve_splits(
    spec: _TaskSpec, args: argparse.Namespace
) -> tuple[str | None, str | None, int]:
    """Settle where the problems and the exemplars come from, and how many.

    The parser cannot carry these defaults because they are not shared: IFEval has one
    split and it is called ``train``, HumanEval's loader takes no split at all, and
    neither takes a few-shot prefix. So the parser defaults to ``None`` -- "the caller
    said nothing" -- and the task decides. An option the task cannot honour is an
    error rather than a silently ignored argument, because a run that quietly dropped
    ``--shots 5`` produces a number that looks like a five-shot score.
    """
    from dynquant.errors import DynQuantError

    if args.split is not None and spec.split is None:
        raise DynQuantError(
            f"--split {args.split!r} was given, but {spec.key} ships a single set and its "
            f"loader takes no split argument. Drop --split."
        )
    split = spec.split if args.split is None else args.split

    if not spec.takes_shots:
        for flag, value in (("--shots", args.shots), ("--shot-split", args.shot_split)):
            if value is not None:
                raise DynQuantError(
                    f"{flag} was given, but {spec.key} takes no few-shot prefix -- the "
                    f"instruction in the prompt is the whole specification, and "
                    f"{spec.key}'s scorer would ignore the exemplars. Drop {flag}."
                )
        return split, None, 0

    shot_split = spec.shot_split if args.shot_split is None else args.shot_split
    return split, shot_split, spec.shots if args.shots is None else args.shots


def _task_kwargs(spec: _TaskSpec, args: argparse.Namespace, shots: list[Any]) -> dict[str, Any]:
    """The arguments only some tasks take.

    Passed by declared capability rather than by task name: a chain of ``if task ==``
    is how a fifth task gets added to the registry and silently never receives its own
    options.
    """
    from dynquant.errors import DynQuantError

    kwargs: dict[str, Any] = {}
    if spec.takes_shots:
        kwargs["shots"] = shots
    if spec.unverifiable:
        kwargs["on_unverifiable"] = args.on_unverifiable
    if spec.executes_code:
        if not args.allow_execution:
            raise DynQuantError(
                f"{spec.key} is scored by running the code the model wrote, so it needs "
                f"--allow-execution. The sandbox is described in "
                f"dynquant.eval._code_exec; read what it does and does not protect "
                f"against before pointing it at an untrusted checkpoint."
            )
        kwargs["allow_execution"] = True
        # Left unset rather than defaulted here: the task module owns the sandbox
        # budget, and a second copy of the number in the CLI is a second thing to
        # forget to change.
        if args.exec_timeout is not None:
            kwargs["timeout"] = args.exec_timeout
        if args.exec_memory_mb is not None:
            kwargs["memory_mb"] = args.exec_memory_mb
    # Its own capability rather than a rider on `executes_code`. The two coincided
    # while only the code tasks took a framing argument, and text2sql takes one
    # without running anything -- reading `--prompt-style` off the sandbox flag
    # would have made the SQL task silently unstyleable.
    if spec.takes_style:
        kwargs["style"] = args.prompt_style
    return kwargs


def _pick_shots(spec: _TaskSpec, count: int, *, seed: int, split: str | None) -> list[Any]:
    """A fixed few-shot prefix, byte-identical across runs and restarts.

    Drawn from a split the evaluation does not score. Using the scored split would
    put the answer to a graded problem in its own prompt for at least one example,
    which is a leak that reads as a small unexplained gain.
    """
    if count == 0:
        return []
    pool = spec.load(split)
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(pool)), count))
    return [pool[i] for i in chosen]


def _pin_experts_dispatch(model: Any, args: argparse.Namespace) -> dict[str, str] | None:
    """Put every arm of a panel on one experts dispatch, and record which one ran.

    Three arms of the same panel arrive here on three different dispatches with nothing
    on the command line saying so. A ``--map-apply pack`` arm has already been moved off
    the default by ``pack_model``, because a packed bank has to be indexed one expert at a
    time; since :mod:`dynquant.runtime.experts` exists that destination is ``dynquant``
    rather than ``eager``, on a transformers with an interface to register into. A
    ``--map-apply encode`` arm is on whatever ``post_init`` picked, which in
    transformers 5.14.1 is ``grouped_mm``. A baseline handed in through ``model=`` has had
    its banks rewritten into per-expert ``Linear`` modules by ``llm-compressor``, so it
    computes the loop whatever the setting says. Only the first of those three was a
    decision.

    That last one used to be written here as "has no dispatch left at all", and it is
    wrong in a way worth keeping written down. ``linearize_moe`` replaces *modules*;
    ``_experts_implementation`` lives on the *config*, which it never touches. Measured on
    a four-layer LFM2-MoE built from this campaign's own config: the attribute reads
    ``grouped_mm`` before and after, the setter is still callable, three banks go to zero
    non-linearised, and this function comes back ``{grouped_mm, eager}`` rather than
    ``None``. So a linearised baseline records the same pair as every other arm and pairs
    with them -- and it does so because of what the config says, not because anything has
    checked that the loop and ``eager`` agree numerically. They should: both index one
    expert at a time. Nothing here has measured it.

    The difference is too large to leave to chance. On LFM2.5-8B-A1B the two dispatches
    disagree on 1.24% of teacher-forced tokens -- 0.29x the effect of quantizing that model
    to 4 bits -- because a top-k router turns a numeric difference into a discrete one and
    22 layers compound it. A panel that varies dispatch alongside quantizer cannot say
    which of the two moved its margin.

    ``eager`` is what this pins to, and half the reason has since stopped being true. It
    was chosen as the only dispatch every arm can share -- a linearised baseline cannot be
    put on ``grouped_mm``, there is nothing left to dispatch -- and defended as also being
    what a downloaded packed checkpoint runs, which is what made an encoder-scored number a
    claim about the artifact. The second half is now false:
    :func:`dynquant.runtime.experts.use_dynquant_experts` serves a packed bank without
    leaving the grouped path, so a downloaded checkpoint runs ``dynquant``, which at this
    model's MoE geometry is bit-identical to ``grouped_mm`` where ``eager`` is not -- 0.00%
    of argmax tokens against 1.95% (``experiments/phase4/probe_experts_dispatch.py``).

    The pin stays on ``eager`` regardless, and the reason it stays is the half that did
    not change: a GPTQ or AWQ arm has been through ``llm-compressor``, which rewrites the
    banks into per-expert ``Linear`` modules, so it computes the indexing loop whatever the
    config says and cannot be moved onto ``grouped_mm`` or onto ``dynquant`` either. There
    is nothing left in it to dispatch. ``eager`` is still the only setting all seven arms
    of a mixed panel can share, and what ``dynquant`` changes is the claim about the
    artifact, not the arithmetic a linearised baseline is stuck with.

    A first draft of this paragraph justified the pin differently -- that the panel's
    already-scored arms ran on ``eager``, so repointing the default would unpair them --
    and that is worth correcting rather than deleting, because the true state is the more
    awkward one. Those arms were scored by a clone predating this function: they carry no
    ``experts`` key at all, and ``_comparability`` treats an absent key as an exemption, so
    they pair with anything. What they actually ran was ``grouped_mm`` for ``bf16`` and for
    the ``encode``-mode DynQuant arms, and the loop for the ``llm-compressor`` ones. The
    panel as banked therefore varies dispatch *alongside* quantizer on exactly the
    comparison it exists to make, and pairing did not object because there was nothing
    recorded to object to. Pinning here is what stops that recurring; the banked arms need
    a re-score, not a defence.

    ``--experts-impl auto`` leaves the model's own choice alone, which is what to run for
    the artifact's own number rather than a cross-arm comparison.

    Returns what the dispatch was when this looked and what it was when this returned, or
    ``None`` for a model whose config carries no such attribute -- a dense model, or one on
    a transformers old enough to have no dispatch to pick. Not a linearised baseline: see
    above. ``found`` is not ``post_init``'s choice on a packed arm: packing has already
    moved it by the time this runs, and the honest report is what was there rather than a
    reconstruction of what would have been. Which is why ``found`` on a packed arm now
    reads ``dynquant`` where it used to read ``eager``, and why that was safe to change:
    pairing keys on ``ran``, which this still sets to ``eager``. Had both fields been in
    ``EXPERTS_PAIRING_FIELDS`` the dispatch rewrite would have unpaired five scored arms
    without altering a number in any of them.
    """
    config = getattr(model, "config", None)
    found = getattr(config, "_experts_implementation", None)
    if found is None:
        return None
    if getattr(args, "experts_impl", "eager") == "eager":
        from dynquant.runtime.linear import use_eager_experts

        use_eager_experts(model)
    ran = getattr(config, "_experts_implementation", None)
    return {"found": str(found), "ran": str(ran)}


def _apply_map(model: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Put the map's widths into the weights, one of the two ways there are.

    Both run the same encoder over the same widths and put the same values into the
    weights; they differ in what the model then holds, and on a MoE in what it then
    computes. Packing keeps the values packed in VRAM, which is the configuration whose
    memory figure is real and the default for that reason. Encoding writes the
    reconstruction back in the compute dtype -- same values, fp16 size -- and is the only
    one that reaches a weight that is not a module.

    The computation diverges because ``pack_model`` moves the experts dispatch off the
    model's default -- to ``dynquant`` where transformers has an interface to register
    into, ``eager`` where it does not -- and the encoder leaves it alone. Those are not the
    same arithmetic on a real model: ``eager`` differs from the default on 1.24% of
    teacher-forced tokens on LFM2.5-8B-A1B, while ``dynquant`` is bit-identical to it.
    :func:`run` pins the dispatch after this returns, which puts both modes back on the
    same footing; a caller reaching this function directly does not get that and has to
    pin its own.

    That last clause is why the choice exists rather than being decided here. On a
    batched-expert MoE the packed runtime has nothing to replace for 91.5% of the
    parameters, and a command that quietly encoded them instead would report a size it
    was not holding. So the caller says which, and the record says which was done.
    """
    if getattr(args, "map_apply", "pack") == "encode":
        return _encode(model, args)
    return _pack(model, args)


def _encode(model: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Write the encoder's output back into the weights, in the compute dtype."""
    from dynquant.quant.quantizer import quantize_model

    bits, metadata = _shared.read_bit_map(args.map, key=args.map_key)
    _shared.check_map_covers(model, bits)
    group_size = int(metadata.get("group_size", args.group_size))

    report = quantize_model(
        model,
        bits,
        group_size=group_size,
        compute_device=getattr(args, "compute_device", "auto"),
        progress=None if args.quiet else _shared.progress_printer("encode"),
    )
    print("\n" + report.summary(), flush=True)
    errors = sorted(layer.relative_error for layer in report.layers.values())
    return {
        "map": args.map,
        "apply": "encode",
        "group_size": group_size,
        "modules": len(report.layers),
        # Named for what it is, not left out. The map's own `nbytes` is the size claim
        # for an encoded arm, and it is in the map file -- what is *not* true is that
        # this model now occupies it, so no key here says so.
        "holds": "compute-dtype values; the size claim is the map's, not this model's",
        "relative_error_median": errors[len(errors) // 2] if errors else None,
        "relative_error_max": errors[-1] if errors else None,
    }


def _pack(model: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Swap the named modules onto the packed runtime, in place."""
    import torch

    from dynquant.runtime.linear import pack_model
    from dynquant.runtime.ops import active_backend

    bits, metadata = _shared.read_bit_map(args.map, key=args.map_key)
    _shared.check_map_covers(model, bits)
    group_size = int(metadata.get("group_size", args.group_size))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    report = pack_model(
        model,
        bits,
        group_size=group_size,
        compute_device=getattr(args, "compute_device", "auto"),
        progress=None if args.quiet else _shared.progress_printer("pack"),
    )
    print("\n" + report.summary(), flush=True)

    backend = active_backend().value
    if backend != "cuda":
        # Not a failure -- the numbers stay correct -- but the fallback dequantizes
        # every weight on every call, so a speed figure from it describes the
        # fallback and not the kernel, and saying so is cheaper than a footnote.
        print(
            f"  note: backend is {backend!r}, not the compiled kernels. Accuracy is "
            f"unaffected; anything timed here is the fallback. Run `dynquant doctor`.",
            flush=True,
        )
    # Two different numbers, and conflating them would misreport the headline claim.
    # Encoding may borrow the GPU even for a CPU-resident model (see
    # `dynquant.quant.device`), so the peak spans a one-time working set that is gone
    # by the time anything runs. What a reader wants from a packed model is what it
    # *holds*, so that is recorded separately and after the transients are released.
    memory: dict[str, int | None] = {"cuda_pack_peak_bytes": None, "cuda_resident_bytes": None}
    if torch.cuda.is_available():
        memory["cuda_pack_peak_bytes"] = int(torch.cuda.max_memory_allocated())
        torch.cuda.empty_cache()
        memory["cuda_resident_bytes"] = int(torch.cuda.memory_allocated())

    return {
        "map": args.map,
        "apply": "pack",
        "group_size": group_size,
        "backend": backend,
        "modules": len(report.modules),
        "tied": len(report.tied),
        "fp16_bytes": report.fp16_bytes,
        "packed_bytes": report.packed_bytes,
        **memory,
    }


def _comparability(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten a record down to the settings a pair has to agree on.

    Three tuples rather than one because the record keeps them in three places. The
    decode budget lives under ``decode``, written by the harness beside the batch size
    and the prompt cap; the prompt style lives under ``detail``, written by the task that
    resolved it. Reading them where they already are, rather than promoting them to the
    top level, is what keeps every record this campaign has already written pairable
    against a new one.

    Missing is not ``None``. ``split`` is legitimately ``None`` for a dataset with a
    single set, so a record that never wrote the field has to be distinguishable from one
    that wrote nothing into it -- otherwise the guard compares absence against absence and
    waves through the mismatch it exists to catch.
    """
    values = {field: record.get(field, _ABSENT) for field in PAIRING_FIELDS}
    for prefix, fields in (
        ("decode", DECODE_PAIRING_FIELDS),
        ("detail", DETAIL_PAIRING_FIELDS),
        ("experts", EXPERTS_PAIRING_FIELDS),
    ):
        block = record.get(prefix)
        for field in fields:
            values[f"{prefix}.{field}"] = (
                block.get(field, _ABSENT) if isinstance(block, dict) else _ABSENT
            )
    return values


def problem_set_difference(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Where two records disagree about *what was asked*, which is what stops a pairing.

    A paired test lines two hit vectors up element-wise, so it needs both runs to have
    been scored over the same items in the same order. Every field in the contract names
    something that would break that -- the task, the backend, the split, the shot count
    and seed, the item limit, the decode budget, the prompt framing -- except one.

    ``experts.ran`` names how an answer was computed, not which question was asked. Two
    arms on different expert dispatches answered the same items in the same order and
    their vectors do pair; what the difference costs is the *reading* of the delta, since
    the two dispatches disagree on 1.24% of teacher-forced tokens on LFM2.5-8B-A1B --
    0.29x what quantizing that model to 4 bits moves. That is a caveat to price on a
    comparison, not a reason to refuse it, and ``panel_table`` prices it per row with a
    mark. Excluded here by name from ``EXPERTS_PAIRING_FIELDS``, so a field added there is
    excluded with it rather than quietly becoming fatal.

    This exists because both callers had started to grow their own copy of the subtraction
    and a third was about to. A caller that wants the difference including the dispatch
    still has ``_comparability``.
    """
    computation = {f"experts.{field}" for field in EXPERTS_PAIRING_FIELDS}
    a, b = _comparability(left), _comparability(right)
    return {key: (a[key], b[key]) for key in a if a[key] != b[key] and key not in computation}


def _compare(record: dict[str, Any], path: str) -> dict[str, Any]:
    """Paired McNemar against a record written by an earlier run."""
    from dynquant.errors import DynQuantError
    from dynquant.eval.compare import compare_paired

    source = Path(path)
    if not source.is_file():
        raise DynQuantError(f"no evaluation record at {source}")
    other = json.loads(source.read_text(encoding="utf-8"))

    theirs = _comparability(other)
    for field, mine in _comparability(record).items():
        if mine is _ABSENT and field not in _OPTIONAL_COMPARABILITY:
            # Not a user error: this run built its own record a moment ago. A field
            # dropped from it would make the guard below compare absence against absence
            # and wave through every mismatch, so the check that the guard can still see
            # has to come first.
            raise DynQuantError(
                f"this run's record has no {field!r} field, so it cannot be checked "
                f"against {source} for comparability. That is a bug in `dynquant eval`, "
                f"not something the command line did wrong."
            )
        if theirs[field] != mine:
            was = "absent" if theirs[field] is _ABSENT else repr(theirs[field])
            used = "nothing" if mine is _ABSENT else repr(mine)
            raise DynQuantError(
                f"{source} was measured with {field}={was} but this run used {used}. "
                f"A paired test needs the same problems in the same order; comparing "
                f"across settings would report a harness difference as a quantization "
                f"effect."
            )
    if len(other.get("hits", [])) != len(record["hits"]):
        raise DynQuantError(
            f"{source} scored {len(other.get('hits', []))} problems, this run scored "
            f"{len(record['hits'])}. Not the same problem set."
        )

    paired = compare_paired(
        record["hits"],
        other["hits"],
        label_a=str(record["label"]),
        label_b=str(other.get("label", source.name)),
    )
    print("\n" + paired.summary(), flush=True)
    return paired.as_dict()
