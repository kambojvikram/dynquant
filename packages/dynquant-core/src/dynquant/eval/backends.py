"""Score a task through a serving engine instead of through ``model.generate``.

Generative evaluation at campaign volume through ``transformers`` is the difference
between a week and a month: ~32 checkpoints times five tasks, each several thousand
greedy generations of a few hundred tokens. vLLM does the same work with continuous
batching and paged attention, and it is also what the checkpoints will actually be
served with, so evaluating through it measures the thing users get.

**The engine is the only thing that changes.** :class:`VllmBackend` implements
:class:`~dynquant.eval.harness.EvalBackend`, which takes prompt token ids and returns
continuation token ids -- so the prompts are built by the task, tokenized by the
harness, detokenized by the harness and cut at stop strings by the harness, exactly as
in the ``transformers`` path. Passing ids rather than strings is load-bearing: vLLM
tokenizes with ``add_special_tokens=True``, so handing it a chat-templated *string*
would give every Llama-3 and Gemma-3 prompt two leading BOS tokens while the
``transformers`` arm got one. That is worth a few points of instruction-following, it
raises no error, and it would land in the results table as a runtime difference.

**This is the offline engine, not an HTTP server.** ``vllm.LLM`` is the same scheduler,
the same paged attention and the same kernels a served deployment runs; what an
OpenAI-compatible endpoint adds is a process boundary, a detokenization step performed
by the *server* rather than by this harness, and a lifecycle to manage. The first is
not needed on a single box, and the second is the guarantee above given away -- the
whole point is that only the engine differs.

Requires vLLM >= 0.5, where ``LLM.generate`` accepts ``TokensPrompt`` mappings. The
import is lazy, so ``dynquant.eval`` stays importable on a box without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dynquant._logging import get_logger
from dynquant.errors import DynQuantError

from .harness import EvalBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from .harness import EvalConfig

__all__ = ["VllmBackend"]

_log = get_logger(__name__)


class VllmBackend(EvalBackend):
    """Greedy generation through an offline vLLM engine.

    ``EvalConfig.batch_size`` does not apply here and is ignored: it is a
    ``transformers``-path knob for how large a padded tensor to build, and vLLM
    schedules its own concurrency from ``max_num_seqs`` and the KV-cache budget.
    Every prompt is submitted in one call, because throttling the scheduler to
    32-request waves is most of what the engine was chosen for.
    """

    name = "vllm"

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    @classmethod
    def from_pretrained(cls, model_path: str, **engine_kwargs: Any) -> VllmBackend:
        """Build the engine. ``engine_kwargs`` go straight to ``vllm.LLM``.

        A DynQuant checkpoint needs ``quantization="dynquant"``, served by
        :mod:`dynquant.integration.vllm_plugin`; GPTQ and AWQ baselines are
        recognised from their own ``config.json`` and need nothing.
        """
        try:
            from vllm import LLM
        except ImportError as exc:  # pragma: no cover -- absent on the dev box
            raise DynQuantError(
                "vLLM is not installed; `pip install vllm` or evaluate through "
                "transformers by passing the model itself to the task"
            ) from exc
        return cls(LLM(model=model_path, **engine_kwargs))

    def generate_ids(
        self,
        prompt_ids: Sequence[Sequence[int]],
        config: EvalConfig,
        *,
        progress: Callable[[int, int], None] | None = None,
        extras: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> list[list[int]]:
        if extras and any(extra is not None for extra in extras):
            raise DynQuantError(
                "these prompts carry encoder inputs and this engine was handed token ids "
                "only. vLLM takes multi-modal inputs through its own `multi_modal_data` "
                "field on the request, not through the processor tensors the transformers "
                "path uses, so forwarding them is not a matter of renaming a key. Score "
                "the audio arm through `--backend transformers`."
            )
        if not prompt_ids:
            return []

        sent = [list(row) for row in prompt_ids]
        outputs = self._engine.generate(
            prompts=[{"prompt_token_ids": row} for row in sent],
            sampling_params=self._sampling_params(config),
            use_tqdm=False,
        )
        ordered = _in_request_order(outputs)
        _assert_aligned(ordered, sent)

        if progress is not None:
            progress(len(sent), len(sent))
        return [[int(i) for i in out.outputs[0].token_ids] for out in ordered]

    def _sampling_params(self, config: EvalConfig) -> Any:
        """Greedy, one sequence, pinned field by field to match the ``transformers`` arm.

        The penalties are set explicitly even though they are already
        :class:`SamplingParams`' defaults, because their being defaults here is exactly
        what made the arms disagree: ``model.generate`` merges the *checkpoint's*
        ``generation_config``, so a checkpoint shipping ``repetition_penalty: 1.1`` --
        Qwen2.5-1.5B-Instruct does -- decoded with a penalty through ``transformers``
        and without one here, for a 23-point gap on five-shot GSM8K. The
        ``transformers`` side is fixed at its root by
        :func:`~dynquant.eval.harness.greedy_generation_config`; naming them here is so
        the two sides can be read against each other, and so a vLLM release that
        changes a default cannot move a score silently.

        Offline ``LLM.generate`` uses a ``SamplingParams`` it is handed as-is, so the
        engine's ``--generation-config auto`` behaviour -- which would otherwise apply
        that same checkpoint file here -- never gets a say. That is why this is pinned
        in the request rather than in the engine's constructor, where the argument
        would also have to exist in every supported vLLM version.

        ``stop`` is passed only when ``early_stop`` is set, for the same reason it is
        optional there: it is a speed setting and cannot move a score, because
        :func:`~dynquant.eval.harness.generate_batched` cuts the decoded text at the
        earliest stop sequence either way.
        """
        from vllm import SamplingParams

        return SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            repetition_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            min_tokens=0,
            max_tokens=config.max_new_tokens,
            stop=list(config.stop_sequences) if config.early_stop else None,
        )


def _in_request_order(outputs: Sequence[Any]) -> list[Any]:
    """Sort completions by request id, falling back to the order returned.

    ``LLM.generate`` is documented to return outputs in submission order, and its
    request counter persists across calls, so the ids of the second call start where
    the first left off rather than at zero -- which is why this sorts by the integer
    rather than checking for a permutation of ``range(n)``. Sorting when the order is
    already right costs nothing; not sorting when it is wrong scores every answer
    against another problem's gold label.

    A backend whose ids are not integers is not wrong, only unsortable, so the
    returned order stands and :func:`_assert_aligned` is what catches a real
    misalignment.
    """
    try:
        keyed = sorted(outputs, key=lambda out: int(out.request_id))
    except (AttributeError, TypeError, ValueError):
        return list(outputs)
    return keyed


def _assert_aligned(outputs: Sequence[Any], sent: Sequence[Sequence[int]]) -> None:
    """Every completion must carry back the prompt it was asked about.

    The direct check, rather than trusting the ordering: engines echo
    ``prompt_token_ids``, so alignment is verifiable rather than assumed. A
    misalignment here is the single most expensive failure this module can have --
    it produces a complete, plausible, entirely meaningless results table, and it
    would be attributed to the runtime under test.
    """
    if len(outputs) != len(sent):
        raise DynQuantError(f"vLLM returned {len(outputs)} completions for {len(sent)} prompts")
    for index, (out, prompt) in enumerate(zip(outputs, sent, strict=True)):
        echoed = getattr(out, "prompt_token_ids", None)
        if echoed is None:  # pragma: no cover -- every vLLM version echoes them
            _log.warning("vLLM output %d carries no prompt_token_ids; alignment unverified", index)
            continue
        if [int(i) for i in echoed] != list(prompt):
            raise DynQuantError(
                f"vLLM completion {index} answers a different prompt than it was sent "
                f"({len(echoed)} ids back, {len(prompt)} sent); results would be scored "
                f"against the wrong gold labels"
            )
