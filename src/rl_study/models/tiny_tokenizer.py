"""A deterministic printable-ASCII tokenizer for offline reasoning tasks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class EncodedBatch:
    input_ids: Tensor
    attention_mask: Tensor


class TinyTokenizer:
    """Character tokenizer with stable IDs and a 128-entry model vocabulary."""

    pad_token = "<pad>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    unk_token = "<unk>"

    def __init__(self) -> None:
        printable_ascii = tuple(chr(code) for code in range(32, 127))
        self._tokens = (
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
            *printable_ascii,
        )
        self._token_to_id = {token: index for index, token in enumerate(self._tokens)}

    @property
    def vocab_size(self) -> int:
        return 128

    @property
    def defined_token_count(self) -> int:
        return len(self._tokens)

    @property
    def pad_token_id(self) -> int:
        return self._token_to_id[self.pad_token]

    @property
    def bos_token_id(self) -> int:
        return self._token_to_id[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        return self._token_to_id[self.eos_token]

    @property
    def unk_token_id(self) -> int:
        return self._token_to_id[self.unk_token]

    def encode(
        self, text: str, *, add_bos: bool = True, add_eos: bool = True
    ) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_token_id)
        ids.extend(
            self._token_to_id.get(character, self.unk_token_id) for character in text
        )
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(
        self, token_ids: Iterable[int], *, skip_special_tokens: bool = True
    ) -> str:
        characters: list[str] = []
        special_ids = {
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
            self.unk_token_id,
        }
        for token_id in token_ids:
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(
                    f"token ID {token_id} is outside vocabulary size {self.vocab_size}"
                )
            if token_id >= self.defined_token_count:
                if not skip_special_tokens:
                    characters.append(self.unk_token)
                continue
            if skip_special_tokens and token_id in special_ids:
                continue
            characters.append(self._tokens[token_id])
        return "".join(characters)

    def batch_encode(
        self,
        texts: Iterable[str],
        *,
        max_length: int,
        add_bos: bool = True,
        add_eos: bool = True,
        truncation: bool = False,
    ) -> EncodedBatch:
        encoded = [
            self.encode(text, add_bos=add_bos, add_eos=add_eos) for text in texts
        ]
        if not encoded:
            raise ValueError("batch_encode requires at least one text")
        longest = max(len(ids) for ids in encoded)
        if longest > max_length and not truncation:
            raise ValueError(
                f"encoded length {longest} exceeds max_length={max_length}"
            )
        target_length = min(longest, max_length)
        input_ids = torch.full(
            (len(encoded), target_length), self.pad_token_id, dtype=torch.int64
        )
        attention_mask = torch.zeros((len(encoded), target_length), dtype=torch.bool)
        for row, ids in enumerate(encoded):
            kept = ids[:target_length]
            if truncation and len(ids) > target_length and add_eos:
                kept[-1] = self.eos_token_id
            input_ids[row, : len(kept)] = torch.tensor(kept, dtype=torch.int64)
            attention_mask[row, : len(kept)] = True
        return EncodedBatch(input_ids=input_ids, attention_mask=attention_mask)
