# RL-study

LLM 강화학습을 수식, 작은 PyTorch 구현, 실제 실행으로 잇는 한국어 중심의
self-contained 강좌입니다. 승인된 [`GOAL.md`](GOAL.md)를 계약으로 삼아
C0~C12 체크포인트와 최종 hosted 검증을 완료했습니다.

> 현재 상태: 20개 완료 조건은 모두 `verified` 또는 근거가 있는
> `external-manual`입니다. Linux/macOS/Windows × Python 3.10/3.12 CI,
> scheduled notebook 감사와 무료 Colab 새 CPU runtime toy 실행까지 통과했습니다.

## 15분 quickstart

Python 3.10~3.12와 CPU만 있으면 됩니다. 아래 demo는 외부 model/API를 받지 않고
DPO, RLHF-PPO, GRPO, DAPO, Agentic RL을 실제 한 update씩 실행합니다.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m rl_study.demo \
  --profile toy --non-interactive --output-dir artifacts/demo --json
```

Windows에서는 `.venv\Scripts\python`을 사용하세요. 매 실행은 기존 결과를
덮어쓰지 않고 새 `artifacts/demo/demo-*` 아래에 다음을 만듭니다.

- `summary.json`: 환경, 실행 시간, 실제 metric과 해석 경계
- `comparison.png`: 같은 fixed prompt의 verifier reward
- `report.html`: JavaScript 없이 읽는 접근 가능한 정적 보고서
- `compare.html`: 계열·reward·KL·entropy·clip fraction filter와 학습 전후 응답
- 5개 checkpoint와 각각의 `experiment-card.json`

Agentic RL은 환경과 metric이 달라 TinyReasoning reward 그래프에 섞지 않습니다.
한 번의 toy update는 논문 규모 재현이나 알고리즘 순위가 아닙니다. 자세한 설치와
artifact 읽기는 [15분 시작](docs/getting-started.md), 장치별 지원은
[하드웨어 표](docs/hardware.md), 오류는 [문제 해결](docs/troubleshooting.md)을
보세요. 날짜·환경·시간·artifact/checkpoint hash를 검증한 로컬 실행은
[C11 demo evidence](docs/research/C11_DEMO_EVIDENCE.json)에 남겼습니다. 최종 후보
commit의 독립 환경 17단계 감사 결과는
[C12 final audit](docs/research/C12_FINAL_AUDIT.json), hosted run/job/artifact는
[C12 hosted audit](docs/research/C12_HOSTED_AUDIT.json)에 있습니다.

현재 offline toy core는 classic RL, DPO, RLHF-PPO, GRPO, DAPO,
RLOO/Dr. GRPO/GSPO와 Agentic RL의 학습·중단 재개·평가를 실행할 수 있습니다.

## 강좌 notebook

처음이면 [한국어 L00 전체 지도](notebooks/ko/L00_rl_map.ipynb)에서 시작하세요.
한국어판 검증 뒤 같은 코드·수식·검사를 가진
[영어 mirror](notebooks/en/L00_rl_map.ipynb)를 생성했습니다.

| 경로 | 순서 | 학습자 예상 시간 |
|---|---|---:|
| 빠른 경로 | L00 → L01 → L03 → L04 → L07 → L08 → L09 → L10 → L11 → L12 → L13 → L15 → L16의 `[필수/CORE]` | 369분(약 6시간) |
| 전체 경로 | L00~L16의 core·deep dive·회상 문제·오답노트 | 845분(약 14시간) |

예상 시간은 읽기·예측·손계산·exercise를 포함합니다. 반면 2026-08-19 M4
macOS arm64, Python 3.10.12, PyTorch 2.13.0 CPU에서 측정한 **코드 실행만의**
합계는 한국어 28.50초, 영어 26.89초였고 최대 process-tree RSS는 각각
528,318,464 byte와 535,986,176 byte였습니다. 실패 시도까지 보존한 원본과
마지막 clean set은 [C10 실행 manifest](docs/research/C10_NOTEBOOK_EXECUTIONS.jsonl),
[C10 요약](docs/research/C10_NOTEBOOK_REPORT.json)에서 확인할 수 있습니다.

```bash
# source/metadata/output, warning, fresh execution count 검증
python scripts/check_notebook_contract.py --language all --require-executed
python scripts/check_bilingual_parity.py

# 각 notebook을 서로 다른 fresh kernel에서 다시 실행
python scripts/execute_notebooks.py --language all --kernel-name rl-study
```

[무료 Colab quickstart](notebooks/colab/RL_study_quickstart.ipynb)는
clone → install → offline toy demo → opt-in 실제 모델 smoke 순서를 고정합니다.
2026-08-20 새 무료 CPU runtime에서 exact commit clone, PyTorch 2.13.0+cpu 설치,
toy demo 5개 update와 최종 assertion을 실행했고 fallback 없이 완료했습니다.
환경·시간·출력 경계는 [C10 Colab evidence](docs/research/C10_COLAB_EVIDENCE.json)에
있습니다. 실제 모델 smoke는 opt-in을 꺼 둔 채 건너뛰었으므로 GPU 성공을
주장하지 않습니다.

## 고정된 범위

- 고전 RL: bandit, MDP, MC/TD/Q-learning, DQN, REINFORCE, actor-critic, GAE, PPO
- LLM RL: reward modeling, RLHF-PPO, DPO, GRPO, DAPO, RLOO, Dr. GRPO, GSPO
- Agentic RL: 완전 오프라인 tool 환경 2개와 선택형 ALFWorld adapter
- 17개 한국어 notebook을 먼저 검증한 뒤 17개 영어 mirror와 Colab을 제공
- `toy` CPU/offline, 실제 공개 모델용 `laptop`, multi-GPU용 `server` profile

전체 웹 문서의 정보 구조는 [MkDocs 홈](docs/index.md), 강좌 선택은
[강좌 지도](docs/course-map.md), 수식은 [수학 최소 도구함](docs/math.md),
선택 비교는 [알고리즘 카드](docs/algorithms/cards.md)에 있습니다.

최신 문헌, exact upstream revision, model/dataset 크기와 라이선스 판정은
[`C1_SOURCE_AUDIT.md`](docs/research/C1_SOURCE_AUDIT.md)와
[`sources.yml`](docs/sources.yml)에 고정되어 있습니다. DAPO 공식 저장소에는
감사 시점에 라이선스가 없어 그 코드나 문서를 복사·변형하지 않습니다.

## 라이선스

새로 작성한 코드와 문서는 [Apache License 2.0](LICENSE)으로 배포합니다.
외부 model, dataset, framework와 benchmark는 각자의 라이선스를 유지하며 이
저장소에 재배포하지 않습니다. 자세한 현재 상태는 [NOTICE](NOTICE)를 보세요.

강좌의 lesson별 목표와 실행 demo는
[`curriculum.md`](docs/design/curriculum.md), notebook의 접근성·한영 parity 계약은
[`notebook-style.md`](docs/design/notebook-style.md), typed API와 재현 artifact
schema는 [`architecture.md`](docs/design/architecture.md)에 있습니다.

RLHF-PPO와 DPO의 수식↔코드 설명은
[`rlhf-ppo.md`](docs/algorithms/rlhf-ppo.md),
[`dpo.md`](docs/algorithms/dpo.md), 동일 초기 weight/prompt budget 비교 결과는
[`C6_ALIGNMENT_BENCHMARK.json`](docs/research/C6_ALIGNMENT_BENCHMARK.json)에서
볼 수 있습니다.

실제 공개 모델 경로는 [`laptop-server.md`](docs/profiles/laptop-server.md)에
있습니다. 승인된 download guard를 거쳐 SmolLM2-135M LoRA를 이 Mac CPU에서
실행했고, 결과와 미실행 범위는
[`C8_LAPTOP_RUN.json`](docs/research/C8_LAPTOP_RUN.json),
[`C8_SERVER_STATUS.json`](docs/research/C8_SERVER_STATUS.json)에 분리했습니다.

```bash
# network 없는 실제 환경 진단
TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv/bin/rl-study preflight \
  --profile laptop --model laptop-smoke --device cpu --json

# model ID/revision/license/bytes를 확인한 뒤에만 download 승인
TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv/bin/rl-study train \
  --config configs/laptop/smollm2_lora_sft.yaml --accept-download --json
```

GRPO 계열의 ratio·baseline·길이 reduction 비교는
[`grpo-family.md`](docs/algorithms/grpo-family.md), DAPO 네 구성요소의 독립 구현과
라이선스 경계는 [`dapo.md`](docs/algorithms/dapo.md), 동일 SFT 시작점에서 실행한
10-way component report는
[`C7_GROUP_BENCHMARK.json`](docs/research/C7_GROUP_BENCHMARK.json)에 있습니다.

두 개의 offline tool 환경, action-token mask, step credit와 실패 모드는
[`agentic-rl.md`](docs/algorithms/agentic-rl.md), 동일 초기 policy로 실행한
outcome-broadcast 대 discounted-return 결과는
[`C9_AGENTIC_BENCHMARK.json`](docs/research/C9_AGENTIC_BENCHMARK.json)에 있습니다.

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv/bin/rl-study train \
  --config configs/toy/agentic_returns.yaml --json
```

## 문서·품질 검증

```bash
python -m pytest
python -m ruff check .
python -m mypy src
python scripts/check_links.py --local
python scripts/check_provenance.py
python scripts/check_notebook_contract.py --language all --require-executed
python scripts/check_bilingual_parity.py
python -m mkdocs build --strict
```

Linux/macOS/Windows CPU matrix와 주기적 network/link/notebook audit는
`.github/workflows`에 선언돼 있습니다. 최종 코드 SHA에서 3개 OS와 Python
3.10/3.12의 6개 CPU job, 정적·문서 job, 수동 dispatch한 scheduled audit가 모두
성공했습니다. run/job ID와 만료 가능한 artifact digest는
[hosted audit](docs/research/C12_HOSTED_AUDIT.json)에 고정했습니다.

기여 전에는 [CONTRIBUTING.md](CONTRIBUTING.md), 행동 규칙은
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), 취약점은 [SECURITY.md](SECURITY.md)를
확인하세요. 출처 재사용 경계는 [provenance 문서](docs/provenance.md), 검증하지
않은 범위는 [알려진 한계](docs/known-limitations.md)에 있습니다. Release 변경은
[CHANGELOG.md](CHANGELOG.md), 인용 metadata는 [CITATION.cff](CITATION.cff)를
사용합니다.
