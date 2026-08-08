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

``--map`` swaps the model onto the packed runtime before scoring, so the weights
stay quantized in VRAM. That is the configuration whose memory figure is real; a
directory written by ``dynquant quantize`` holds quantized *values* in the compute
dtype, which gives the right accuracy and the wrong size. So ``--map`` takes the
*unquantized* model plus a bit map, and the two paths agree on accuracy because
they run the same encoder over the same widths.
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

__all__ = ["PAIRING_FIELDS", "TASKS", "run"]

#: Everything two records must agree on before their hit vectors may be paired.
#:
#: ``backend`` is here even though G4 measures the two runtimes as equivalent in
#: *score*. Equal totals are not identical per-item outcomes, and McNemar reads exactly
#: the items the two arms disagree on -- so a cross-backend pair would put engine
#: disagreement into the one cell the test counts.
PAIRING_FIELDS = ("task", "backend", "split", "shots", "shot_seed", "limit")


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


def run(args: argparse.Namespace) -> int:
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

    model, packed = _load_runtime(args, config)
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
        return model, (_pack(model, args) if args.map is not None else None)

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
        "group_size": group_size,
        "backend": backend,
        "modules": len(report.modules),
        "tied": len(report.tied),
        "fp16_bytes": report.fp16_bytes,
        "packed_bytes": report.packed_bytes,
        **memory,
    }


def _compare(record: dict[str, Any], path: str) -> dict[str, Any]:
    """Paired McNemar against a record written by an earlier run."""
    from dynquant.errors import DynQuantError
    from dynquant.eval.compare import compare_paired

    source = Path(path)
    if not source.is_file():
        raise DynQuantError(f"no evaluation record at {source}")
    other = json.loads(source.read_text(encoding="utf-8"))

    for field in PAIRING_FIELDS:
        if field not in record:
            # Not a user error: this run built its own record a moment ago. A field
            # dropped from it would make the guard below compare None against None and
            # wave through every mismatch, so the check that the guard can still see
            # has to come first.
            raise DynQuantError(
                f"this run's record has no {field!r} field, so it cannot be checked "
                f"against {source} for comparability. That is a bug in `dynquant eval`, "
                f"not something the command line did wrong."
            )
        if other.get(field) != record.get(field):
            raise DynQuantError(
                f"{source} was measured with {field}={other.get(field)!r} but this run "
                f"used {record.get(field)!r}. A paired test needs the same problems in "
                f"the same order; comparing across settings would report a harness "
                f"difference as a quantization effect."
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
