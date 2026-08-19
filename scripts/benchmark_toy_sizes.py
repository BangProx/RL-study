"""C2 전용 tiny causal LM 크기 측정 도구.

Model/data download 없이 후보 크기의 CPU/MPS forward+backward 시간을 JSON으로
출력한다. 이 파일의 결과는 설계값을 정하기 위한 근거이며 성능 benchmark나
paper reproduction으로 해석하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Candidate:
    name: str
    vocab_size: int
    sequence_length: int
    batch_size: int
    hidden_size: int
    num_heads: int
    num_layers: int
    intermediate_size: int


CANDIDATES = (
    Candidate("micro", 64, 32, 16, 32, 4, 2, 64),
    Candidate("base", 96, 64, 16, 64, 4, 2, 128),
    Candidate("learning", 128, 64, 16, 96, 4, 3, 192),
)


class TinyCausalLM(nn.Module):
    def __init__(self, config: Candidate) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.sequence_length, config.hidden_size
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.intermediate_size,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, config.num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        _, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        return self.lm_head(self.norm(self.blocks(hidden, mask=causal_mask)))


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(
    candidate: Candidate, device: torch.device, steps: int
) -> dict[str, object]:
    torch.manual_seed(42)
    model = TinyCausalLM(candidate).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    input_ids = torch.randint(
        candidate.vocab_size,
        (candidate.batch_size, candidate.sequence_length),
        device=device,
    )
    targets = torch.roll(input_ids, shifts=-1, dims=1)

    def train_step() -> float:
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, candidate.vocab_size),
            targets[:, :-1].reshape(-1),
        )
        loss.backward()
        optimizer.step()
        synchronize(device)
        return float(loss.detach().cpu())

    for _ in range(3):
        train_step()

    durations: list[float] = []
    losses: list[float] = []
    for _ in range(steps):
        start = time.perf_counter()
        losses.append(train_step())
        durations.append((time.perf_counter() - start) * 1_000)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "candidate": asdict(candidate),
        "device": str(device),
        "parameters": parameter_count,
        "parameter_bytes_fp32": parameter_count * 4,
        "median_train_step_ms": round(statistics.median(durations), 3),
        "p95_train_step_ms": round(sorted(durations)[int(0.95 * (steps - 1))], 3),
        "estimated_100_steps_seconds": round(statistics.median(durations) / 10, 3),
        "first_measured_loss": round(losses[0], 6),
        "last_measured_loss": round(losses[-1], 6),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--steps", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 5:
        raise SystemExit("--steps는 안정적인 중앙값을 위해 5 이상이어야 합니다.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit(
            "MPS를 요청했지만 torch.backends.mps.is_available()이 False입니다."
        )
    device = torch.device(args.device)
    payload = {
        "purpose": "C2 toy-size selection; not a scientific performance benchmark",
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": args.device,
        "steps": args.steps,
        "results": [measure(candidate, device, args.steps) for candidate in CANDIDATES],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
