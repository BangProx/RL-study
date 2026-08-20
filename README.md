# RL-study

[English](README.en.md) · **한국어**

[![CI](https://github.com/BangProx/RL-study/actions/workflows/ci.yml/badge.svg)](https://github.com/BangProx/RL-study/actions/workflows/ci.yml)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-3776AB.svg)](pyproject.toml)
[![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-4c1.svg)](LICENSE)

**LLM 강화학습을 읽는 데서 끝내지 마세요.** GPU나 외부 API 없이 CPU에서
PPO·RLHF·DPO·GRPO·DAPO와 Agentic RL의 핵심 수식을 작은 PyTorch 구현과 실제
update로 연결합니다.

> **처음 시작하기 → [L00 · LLM RL 전체 지도](notebooks/ko/L00_rl_map.ipynb)**

Python과 tensor 기초는 알지만 RL은 처음인 학습자, 논문 수식과 실제 코드 사이가
끊기는 연구자·개발자, 짧은 학습 단위와 즉시 실행 결과가 필요한 학습자를 위한
self-contained 강좌입니다. 설치 없이 먼저 확인하려면
[무료 Colab quickstart](notebooks/colab/RL_study_quickstart.ipynb)를 사용하세요.

## 끝나면 무엇을 할 수 있나요?

- Bellman target부터 PPO clipped objective까지 직접 계산하고 구현합니다.
- LLM의 prompt와 response 중 어느 token이 policy action인지 추적합니다.
- RLHF-PPO의 old/reference/reward/value model 역할을 코드에서 구분합니다.
- DPO·GRPO·DAPO가 데이터, baseline, ratio, reduction을 어떻게 바꾸는지 설명합니다.
- multi-turn tool trajectory의 action mask와 outcome/process credit을 구현합니다.
- toy 실험을 학습·중단 재개·평가하고, laptop/server profile로 확장합니다.

## 먼저 결과를 보세요

아래 그림은 꾸민 예시가 아니라 `python -m rl_study.demo`가 CPU에서 DPO,
RLHF-PPO, GRPO, DAPO를 각각 **한 번 실제 update**한 뒤 같은 TinyReasoning
prompt에서 만든 진단 결과입니다.

![DPO, RLHF-PPO, GRPO, DAPO의 one-update reward, KL, entropy, clip fraction 진단](docs/assets/one-update-diagnostics.png)

한 번의 toy update는 논문 규모 성능이나 알고리즘 순위가 아닙니다. 이 그림의
목적은 reward·KL·entropy·clip fraction이 코드에서 실제로 계산되고 서로 다른
경계를 가진다는 사실을 눈으로 확인하는 것입니다. 그림은
`python scripts/render_readme_asset.py`로 다시 만들 수 있습니다.

## 15분 quickstart

Python 3.10~3.12와 CPU만 있으면 됩니다. demo는 외부 model이나 API를 받지 않고
DPO, RLHF-PPO, GRPO, DAPO, Agentic REINFORCE를 한 update씩 실행합니다.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m rl_study.demo \
  --profile toy --non-interactive --output-dir artifacts/demo --json
```

Windows에서는 `.venv\Scripts\python`을 사용하세요. 실행할 때마다 기존 결과를
덮어쓰지 않고 새 `artifacts/demo/demo-*` 아래에 다음을 만듭니다.

- `summary.json`: 환경, 실행 시간, 실제 metric과 해석 경계
- `comparison.png`: 동일 prompt의 reward·KL·entropy·clip fraction
- `report.html`: JavaScript 없이 읽을 수 있는 정적 보고서
- `compare.html`: 지표 filter와 학습 전후 응답 비교
- checkpoint 5개와 각각의 `experiment-card.json`

설치·결과 읽기는 [15분 시작](docs/getting-started.md), 장치별 지원은
[하드웨어 표](docs/hardware.md), 오류는 [문제 해결](docs/troubleshooting.md)을
보세요.

## 두 학습 경로

| 경로 | 순서 | 학습자 예상 시간 |
|---|---|---:|
| 빠른 경로 | L00 → L01 → L03 → L04 → L07 → L08 → L09 → L10 → L11 → L12 → L13 → L15 → L16의 `[필수/CORE]` | 369분, 약 6시간 |
| 전체 경로 | L00~L16의 core·deep dive·회상 문제·오답노트 | 845분, 약 14시간 |

시간은 읽기·예측·손계산·exercise를 포함합니다. 2026-08-19 M4 macOS arm64,
Python 3.10.12, PyTorch 2.13.0 CPU에서 측정한 **코드 실행만의** 합계는 한국어
28.50초, 영어 26.89초였습니다. 전체 순서와 lesson별 질문은
[강좌 지도](docs/course-map.md)에 있습니다.

## 알고리즘 차이를 한 화면에서

![PPO, DPO, GRPO, DAPO의 데이터, baseline, 목적함수, update 경계 비교](docs/assets/alignment-loss-map.svg)

| 범위 | 포함 내용 |
|---|---|
| 고전 RL | bandit, MDP, MC/TD/Q-learning, DQN, REINFORCE, actor-critic, GAE, PPO |
| LLM RL | reward modeling, RLHF-PPO, DPO, GRPO, DAPO, RLOO, Dr. GRPO, GSPO |
| Agentic RL | 오프라인 tool 환경 2개, action-token mask, multi-step credit |
| 실행 profile | CPU/offline `toy`, 공개 모델 `laptop`, multi-GPU `server` |

수식과 구현 선택은 [알고리즘 카드](docs/algorithms/cards.md)에서 시작하세요.
각 lesson과 상세 문서는 서로 왕복 링크되어 있습니다.

## 실제 모델과 서버로 확장

[Laptop·server 가이드](docs/profiles/laptop-server.md)는 download guard를 거쳐
SmolLM2-135M LoRA를 실행하는 laptop preset과, 실행하지 않은 GPU 결과를
`external-manual`로 남기는 server recipe를 분리합니다.

```bash
# network 없이 실제 환경 진단
TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv/bin/rl-study preflight \
  --profile laptop --model laptop-smoke --device cpu --json

# model ID·revision·license·bytes를 확인한 뒤에만 다운로드 승인
TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv/bin/rl-study train \
  --config configs/laptop/smollm2_lora_sft.yaml --accept-download --json
```

## 재현성과 한계

Toy 결과는 수식, tensor shape, gradient, mask와 실행 lifecycle을 검증하는 sanity
check입니다. 논문 규모 benchmark나 알고리즘 순위가 아닙니다. 지원하지 않거나
실행하지 않은 범위는 [알려진 한계](docs/known-limitations.md)에 있습니다.

논문·공식 구현·라이선스는 [출처 원장](docs/provenance.md), 실행 근거를 읽는 법은
[research index](docs/research/README.md)에 정리했습니다. DAPO 공식 저장소는 감사
시점에 명시 라이선스가 없어 코드나 문서를 복사하지 않고 논문 기반 clean-room으로
구현했습니다.

## 기여와 라이선스

기여 전에는 [CONTRIBUTING.md](CONTRIBUTING.md), 행동 규칙은
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), 취약점은 [SECURITY.md](SECURITY.md)를
확인하세요. 이 저장소의 새 코드와 문서는 Apache-2.0이며, 외부 자료의 경계는
[NOTICE](NOTICE), 인용 정보는 [CITATION.cff](CITATION.cff)에 있습니다.
