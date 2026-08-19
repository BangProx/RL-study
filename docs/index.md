# RL-study

LLM 강화학습을 **수식 → 작은 tensor → PyTorch 구현 → 실제 artifact** 순서로
배우는 한국어 중심 강좌입니다. 외부 API와 GPU가 없어도 bandit부터 DAPO,
multi-turn Agentic RL까지 핵심 학습 루프를 CPU에서 실행할 수 있습니다.

## 지금 무엇을 하면 되나요?

처음 방문했다면 다음 세 단계만 하세요.

1. [15분 시작](getting-started.md)에서 설치하고 실제 비교 report를 만듭니다.
2. [강좌 지도](course-map.md)에서 빠른 6시간 경로 또는 전체 14시간 경로를
   고릅니다.
3. 수식이 막힐 때 [수학 최소 도구함](math.md), 용어가 섞일 때
   [용어집](glossary.md)으로 돌아옵니다.

## 이 저장소가 보장하는 것

- 17개 한국어 notebook과 같은 학습 계약을 가진 17개 영어 mirror
- network 없는 `toy` profile과 체크포인트·resume·eval CLI
- Q-learning, DQN, REINFORCE, actor-critic, GAE, PPO, reward model,
  RLHF-PPO, DPO, GRPO, DAPO의 clean-room reference 구현
- RLOO, Dr. GRPO, GSPO 비교와 두 개의 오프라인 Agentic tool 환경
- 실제 공개 LM LoRA laptop smoke와 실행하지 않은 server 결과를 구분한 recipe
- 논문 식, 코드 함수, test, 실행 evidence를 잇는 추적 원장

## 이 저장소가 주장하지 않는 것

짧은 toy 실행은 논문 규모 benchmark 재현이나 알고리즘 순위가 아닙니다. 대형
GPU 결과, hosted CI, Colab 결과는 실제 실행 증거가 있을 때만 통과로 표시합니다.
`paper_reported`, `upstream_reported`, `local_executed`를 섞지 않습니다.

## ADHD 친화적 학습 리듬

각 lesson은 5~8분 micro-section, 먼저 답을 예상하는 칸, 바로 실행하는 작은 실험,
흔한 실수와 회상 문제를 반복합니다. 한 세션에 모든 셀을 끝낼 필요가 없습니다.
`[필수/CORE]`만 따라가도 개념 지도가 끊기지 않도록 설계했습니다.

다음: [15분 시작하기](getting-started.md)
