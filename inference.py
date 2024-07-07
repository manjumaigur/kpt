import argparse
import json
import os

import torch
from torch.nn import functional as F

from tokenizer import Tokenizer
from model import KPT, KPTConfig


torch.manual_seed(6)
torch.cuda.manual_seed(6)


def run(
    input_ids: torch.Tensor,
    tokenizer: Tokenizer,
    model: KPT,
    max_length: int = 100,
    num_sequences: int = 5,
) -> None:
    # TODO: stop at end-token
    tokens = input_ids.unsqueeze(0).repeat(num_sequences, 1)
    while tokens.size(-1) < max_length:
        with torch.no_grad():
            # batch_size, context_length, vocab_size
            logits = model(tokens)
            # take logits at the last position
            # batch_size, vocab_size
            logits = logits[:, -1, :]
            # get probabilities
            probs = F.softmax(logits, dim=-1)
            # top-k sampling of 50
            # 5, 50 & 5, 50
            topk_probs, topk_indices = torch.topk(probs, 100, dim=-1)

            # Sample from the updated probabilities
            sampled_token = torch.multinomial(topk_probs, 1)
            # batch_size, 1
            tok_col = torch.gather(topk_indices, -1, sampled_token)

            tokens = torch.cat((tokens, tok_col), dim=1)

    print("input > ", tokenizer.decode(input_ids.tolist()))
    for i in range(num_sequences):
        _tokens = tokens[i, :max_length].tolist()
        decoded = tokenizer.decode(_tokens)
        print(">", decoded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, help="path to model checkpoint", required=True
    )
    parser.add_argument("--input", type=str, help="input text", required=True)
    parser.add_argument(
        "--max-length",
        type=int,
        help="total generation tokens including input tokens",
        required=True,
        default=100,
    )
    args = parser.parse_args()
    input_text = args.input.strip()
    max_length = args.max_length
    checkpoint_path = args.checkpoint

    if input_text == "":
        raise ValueError("Input cannot be empty")

    if max_length < 0:
        raise ValueError("max_length cannot be negative")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("Checkpoint not found")

    tokenizer_bpe_file_path = "kpt50k.bpe"
    if not os.path.isfile(tokenizer_bpe_file_path):
        raise FileNotFoundError("Tokenizer bpe file not found")

    model_configs = None
    with open("config.json", "r") as f:
        model_configs = json.load(f)["small"]

    model_configs = KPTConfig(**model_configs)
    model = KPT(model_configs)
    # TODO: add an argument for device
    model.to("cuda")
    model = torch.compile(model)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])

    tokenizer = Tokenizer(tokenizer_bpe_file_path)

    input_ids = tokenizer.encode(input_text)

    input_ids = torch.tensor(input_ids, dtype=torch.long, device="cuda")

    run(input_ids, tokenizer, model, max_length)
