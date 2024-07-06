import argparse
import json
import os

import torch
from torch.nn import functional as F

from tokenizer import Tokenizer
from model import KPT


def run(
    input_ids: torch.Tensor, tokenizer: Tokenizer, model: KPT, max_length: int = 100
) -> None:
    while input_ids.size(1) < max_length:
        with torch.no_grad():
            # batch_size, context_length, vocab_size
            logits, _ = model(input_ids)
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

            input_ids = torch.cat((input_ids, tok_col), dim=1)

    print(tokenizer.decode(input_ids.tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="input text", required=True)
    parser.add_argument(
        "--max-length",
        type=int,
        help="total generation tokens including input tokens",
        required=True,
        default=100,
    )
    args = parser.parse_args()
    input_text = args.input.trim()
    max_length = args.max_length

    if input_text == "":
        raise ValueError("Input cannot be empty")

    if max_length < 0:
        raise ValueError("max_length cannot be negative")

    tokenizer_bpe_file_path = "kpt50k.bpe"
    if not os.path.isfile(tokenizer_bpe_file_path):
        raise FileNotFoundError("Tokenizer bpe file not found")

    tokenizer = Tokenizer(tokenizer_bpe_file_path)
    input_ids = tokenizer.encode(input_text)

    # TODO: add an argument for device
    input_ids = torch.tensor(input_ids, dtype=torch.long, device="cuda")

    model_configs = None
    with open("config.json", "r") as f:
        model_configs = json.load(f)["small"]

    model = KPT(model_configs)
    model.to("cuda")
    model = torch.compile(model)
    checkpoint = torch.load("model_12300.pt")
    model.load_state_dict(checkpoint["model"])

    run(input_ids, tokenizer, model, max_length)
