# 강좌 지도

큰 질문에서 시작해 필요한 수학과 구현 디테일로 내려갑니다. 각 lesson은 이전
notebook의 kernel 상태에 의존하지 않습니다.

```text
왜 RL인가?
  ├─ 확률·gradient ── RL 언어 ── 가치 기반 ── policy gradient ── PPO
  └─ LLM policy ── preference/reward ── RLHF·DPO ── GRPO·DAPO
                                                └─ Agentic RL
```

## 두 가지 경로

| 경로 | 추천 대상 | 순서 | 예상 학습 시간 |
|---|---|---|---:|
| 빠른 | 전체 지도를 먼저 잡고 싶은 사람 | 아래 ★ lesson의 `[필수/CORE]` | 369분, 약 6시간 |
| 전체 | 수식·구현·오답까지 직접 확인할 사람 | L00~L16 전체 | 845분, 약 14시간 |

시간은 읽기, 예측, 손계산, exercise를 포함한 학습자 예상치입니다. notebook의
실제 코드 실행 시간과는 다른 값입니다.

## L00~L16

| Lesson | 핵심 질문 | 빠른 경로 |
|---|---|:---:|
| L00 | 전체 지형에서 지금 어디인가? | ★ |
| L01 | 확률·log-prob·entropy·KL은 무엇인가? | ★ |
| L02 | autodiff와 optimizer는 어떻게 parameter를 바꾸나? |  |
| L03 | state/action/reward/return/policy는 어떻게 연결되나? | ★ |
| L04 | Bellman 식이 value iteration 코드가 되는 과정은? | ★ |
| L05 | MC·TD·Q-learning의 target 차이는? |  |
| L06 | DQN의 replay와 target network가 왜 필요한가? |  |
| L07 | REINFORCE와 baseline은 왜 작동하나? | ★ |
| L08 | actor-critic·GAE·PPO는 variance와 update를 어떻게 제어하나? | ★ |
| L09 | LLM에서 token action과 mask는 무엇인가? | ★ |
| L10 | SFT·preference·reward model·verifier는 무엇이 다른가? | ★ |
| L11 | RLHF-PPO는 old/reference/value/reward를 왜 나누나? | ★ |
| L12 | DPO는 preference를 어떤 분류 loss로 바꾸나? | ★ |
| L13 | GRPO 계열과 DAPO는 PPO의 무엇을 바꾸나? | ★ |
| L14 | toy에서 실제 공개 LM·server로 어떻게 확장하나? |  |
| L15 | multi-turn tool trajectory의 mask와 credit은 무엇이 다른가? | ★ |
| L16 | 실패 모드·평가·재현성을 어떻게 감사하나? | ★ |

lesson별 objective, prerequisite, demo, exercise와 source ID의 완전한 표는
[교육 설계 원장](design/curriculum.md)에 있습니다.

## 막혔을 때 돌아올 곳

- 기호가 낯설다 → [수학 최소 도구함](math.md)
- 단어가 섞인다 → [용어집](glossary.md)
- 알고리즘 선택이 헷갈린다 → [알고리즘 카드](algorithms/cards.md)
- 코드가 실행되지 않는다 → [문제 해결](troubleshooting.md)
