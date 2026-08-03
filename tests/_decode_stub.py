"""A model and tokenizer that decode without a GPU, or weights, or transformers.

The generation loop has failure modes that do not need a real model to expose, and
that are close to impossible to catch *with* one: a batch padded on the wrong side
still returns fluent text, and a prompt truncated from the wrong end still returns an
answer. Both come back as a plausible number a few points low. Exercising the loop
against a stub that reports exactly which tokens reached the model turns "the score
looked a bit low" into an assertion.

The tokenizer is word-per-token with a stable vocabulary, so a decoded generation is
comparable to the string that went in and truncation can be asserted on words rather
than on ids.
"""

from __future__ import annotations

from typing import Any

import torch

PAD_ID = 0


class StubBatch(dict):  # type: ignore[type-arg]
    """What ``tokenizer(...)`` returns: a mapping that also knows ``.to(device)``."""

    def to(self, device: Any) -> StubBatch:
        return self


class StubTokenizer:
    """Word-per-token, with a stable vocabulary.

    ``padding_side`` and ``truncation_side`` are honoured rather than ignored, which
    is what lets a test set them *wrong* and assert the harness pads and truncates
    correctly anyway -- it does both itself, so a tokenizer arriving from a training
    pipeline with its own settings cannot change a score.
    """

    def __init__(self) -> None:
        self.padding_side = "right"
        self.truncation_side = "right"
        self.pad_token_id: int | None = PAD_ID
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.eos_token_id = PAD_ID
        self._ids: dict[str, int] = {}
        self._words: dict[int, str] = {PAD_ID: ""}

    def words_for(self, ids: Any) -> list[str]:
        """Ids back to words, pads included, for asserting what a batch contained."""
        return [self._words[int(i)] for i in ids]

    def _id(self, word: str) -> int:
        if word not in self._ids:
            index = len(self._ids) + 1
            self._ids[word] = index
            self._words[index] = word
        return self._ids[word]

    def encode_words(self, text: str) -> list[int]:
        return [self._id(word) for word in text.split()]

    def __call__(
        self,
        text: str | list[str],
        *,
        return_tensors: str | None = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> Any:
        texts = [text] if isinstance(text, str) else list(text)
        rows = [self.encode_words(item) for item in texts]

        if truncation and max_length is not None:
            rows = [
                row[-max_length:] if self.truncation_side == "left" else row[:max_length]
                for row in rows
            ]
        if padding and rows:
            width = max(len(row) for row in rows)
            rows = [
                [PAD_ID] * (width - len(row)) + row
                if self.padding_side == "left"
                else row + [PAD_ID] * (width - len(row))
                for row in rows
            ]

        if return_tensors != "pt":
            return {"input_ids": rows}

        return StubBatch(
            input_ids=torch.tensor(rows),
            attention_mask=torch.tensor([[int(i != PAD_ID) for i in row] for row in rows]),
        )

    def batch_decode(self, rows: Any, *, skip_special_tokens: bool = False) -> list[str]:
        return [
            " ".join(self._words[int(i)] for i in row if int(i) != PAD_ID).strip() for row in rows
        ]


class StubModel:
    """Appends a fixed reply to whatever it is given.

    A fixed reply is the point: the interesting variable is the *prompt* the loop
    constructed, and a stub that answered differently per prompt would let a
    prompt-construction bug hide behind a plausible answer.
    """

    def __init__(self, tokenizer: StubTokenizer, reply: str) -> None:
        self._tokenizer = tokenizer
        self._reply = tokenizer.encode_words(reply)
        self.training = False
        self.calls = 0
        self.fed: list[list[str]] = []
        """One entry per sequence the model was actually given, as words, pads and
        all. This is the ground truth for "did the model see the question or the
        padding?" -- recorded here rather than on the tokenizer because the harness
        builds the batch itself, so the tokenizer never sees the padded form."""

    def parameters(self) -> Any:
        yield torch.zeros(1)

    def eval(self) -> StubModel:
        self.training = False
        return self

    def train(self) -> StubModel:
        self.training = True
        return self

    def generate(self, *, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.calls += 1
        self.fed.extend(self._tokenizer.words_for(row) for row in input_ids.tolist())
        reply = torch.tensor([self._reply] * input_ids.shape[0], dtype=input_ids.dtype)
        return torch.cat([input_ids, reply], dim=1)
