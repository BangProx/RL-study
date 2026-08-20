# RL-study

[English course map](course-map.en.md) · **한국어 문서**

LLM 강화학습을 **수식 → 작은 tensor → PyTorch 구현 → 실제 artifact** 순서로
배우는 self-contained 강좌입니다. 외부 API와 GPU가 없어도 bandit부터 DAPO,
multi-turn Agentic RL까지 핵심 학습 루프를 CPU에서 실행할 수 있습니다.

> **처음 시작하기 → [15분 시작](getting-started.md)**

Python과 기본 tensor 연산은 알지만 RL이 처음인 학습자, 논문 수식과 코드 사이를
직접 실행하며 연결하고 싶은 연구자·개발자에게 맞습니다.

## 이 강좌를 끝내면

- Bellman target, policy gradient, GAE와 PPO clipping을 직접 계산합니다.
- LLM response token의 log-prob, action mask와 reference KL을 추적합니다.
- RLHF-PPO, DPO, GRPO, DAPO의 데이터와 objective 차이를 구현으로 설명합니다.
- multi-turn tool trajectory의 action과 tool output을 분리하고 credit을 배분합니다.
- 학습을 checkpoint에서 재개하고 metric·artifact·실행 조건을 함께 검사합니다.

## 세 단계 학습 동선

1. [15분 시작](getting-started.md)에서 실제 비교 report를 만듭니다.
2. [강좌 지도](course-map.md)에서 빠른 6시간 또는 전체 14시간 경로를 고릅니다.
3. 수식이 막히면 [수학 최소 도구함](math.md), 알고리즘이 섞이면
   [한눈에 비교](algorithms/cards.md)로 돌아옵니다.

## 제공하는 것

- 17개 한국어 notebook과 같은 코드·수식을 쓰는 17개 영어 mirror
- network 없는 `toy` profile과 checkpoint·resume·eval CLI
- Q-learning부터 RLHF-PPO, DPO, GRPO, DAPO까지 clean-room reference 구현
- RLOO, Dr. GRPO, GSPO 비교와 두 개의 오프라인 Agentic tool 환경
- 실제 공개 LM LoRA laptop smoke와 실행하지 않은 server 결과를 나눈 recipe
- 논문 식, 코드 함수, test와 실행 evidence를 연결하는 provenance

## ADHD 친화적 학습 리듬

각 lesson은 5~8분 micro-section, 먼저 답을 예상하는 칸, 바로 실행하는 작은 실험,
흔한 실수와 회상 문제를 반복합니다. 한 세션에 모든 셀을 끝낼 필요가 없고,
`[필수/CORE]`만 따라가도 개념 지도가 끊기지 않습니다.

## 결과를 해석하는 경계

짧은 toy 실행은 논문 규모 benchmark 재현이나 알고리즘 순위가 아닙니다. 대형 GPU
결과와 외부 benchmark는 실제 실행 증거가 있을 때만 통과로 표시합니다.
`paper_reported`, `upstream_reported`, `local_executed`를 구분하는 상세 원칙은
[provenance](provenance.md)에서 확인하세요.

다음: [15분 시작하기](getting-started.md)
