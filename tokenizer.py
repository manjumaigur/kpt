from typing import List

import tiktoken
from tiktoken.load import load_tiktoken_bpe


class Tokenizer:
    def __init__(self, bpe_file: str) -> None:
        mergeable_ranks = load_tiktoken_bpe(bpe_file)
        self.special_tokens = {
            "<unk>": 50254,
            "<s>": 50255,
            "</s>": 50256,
        }
        # GPT-4o split pattern
        pat_str = "|".join(
            [
                r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
                r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
                r"""\p{N}{1,3}""",
                r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
                r"""\s*[\r\n]+""",
                r"""\s+(?!\S)""",
                r"""\s+""",
            ]
        )
        explicit_n_vocab = 50257
        self.encoder = tiktoken.core.Encoding(
            name="kpt50k",
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
            explicit_n_vocab=explicit_n_vocab,
        )

    def encode(self, text: str) -> List[int]:
        # <unk> is allowed while encoding as the dataset[1] was pre-tokenized and contains lot of <unk>
        # [1]: https://huggingface.co/datasets/ai4bharat/sangraha
        return self.encoder.encode(
            text, allowed_special={"<unk>"}, disallowed_special={"<s>, </s>"}
        )

    def decode(self, tokens: List[int]) -> str:
        return self.encoder.decode(tokens)

    def _visualise_tokens(self, token_values: List[bytes]) -> None:
        background = [f"\u001b[48;5;{i}m" for i in [167, 179, 185, 77, 80, 68, 134]]

        unicode_token_values = [
            x.decode("utf-8", errors="replace") for x in token_values
        ]

        running_length = 0
        last_color = None
        for token in unicode_token_values:
            color = background[running_length % len(background)]
            if color == last_color:
                color = background[(running_length + 1) % len(background)]
                assert color != last_color
            last_color = color
            running_length += len(token)
            print(color + token, end="")
        print("\u001b[0m")

    def visualize(self, text: str) -> None:
        tokens = self.encode(text)
        self._visualise_tokens(self.encoder.decode_tokens_bytes(tokens))
