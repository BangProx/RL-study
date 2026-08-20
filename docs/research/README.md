# 실행 근거 읽기

이 폴더는 강좌의 주장과 실제 실행을 연결합니다. JSON 숫자는 순위표가 아니라
**어떤 환경에서 무엇을 실행했고 어디까지 해석할 수 있는지** 남기는 근거입니다.

> English: These records connect course claims to local executions. They are
> reproducibility evidence, not an algorithm leaderboard.

## C1~C9 지도

| 단계 | 파일 | 확인하는 것 |
|---|---|---|
| C1 | [문헌·라이선스 감사](C1_SOURCE_AUDIT.md) | 논문, 공식 구현, exact revision, license와 clean-room 경계 |
| C2 | [Toy 크기 측정](C2_TOY_BENCHMARK.json) | laptop CPU에서 model 크기별 train-step 시간과 선택 근거 |
| C3 | 별도 JSON 없음 | package, 수학, config, checkpoint 기반을 unit/integration test로 검증한 단계 |
| C4 | [고전 RL 실행](C4_CLASSIC_BENCHMARK.json) | DP, MC/TD/Q-learning, DQN, policy gradient와 PPO sanity check |
| C5 | [LLM 기반 실행](C5_LLM_FOUNDATION_BENCHMARK.json) | SFT, reward model, mask, held-out 진단과 shortcut failure |
| C6 | [DPO·RLHF 비교](C6_ALIGNMENT_BENCHMARK.json) | 동일 SFT 시작점과 prompt budget에서 alignment 경로 비교 |
| C7 | [GRPO 계열 비교](C7_GROUP_BENCHMARK.json) | GRPO, DAPO, RLOO, Dr. GRPO, GSPO 구성요소와 budget |
| C8 | [Laptop 실행](C8_LAPTOP_RUN.json) · [Server 상태](C8_SERVER_STATUS.json) | 실제 소형 LM 실행과 미실행 분산 recipe의 분리 |
| C9 | [Agentic RL 실행](C9_AGENTIC_BENCHMARK.json) | multi-turn tool trajectory의 broadcast와 step return 비교 |

C3는 성능 측정 단계가 아니라 이후 모든 실험이 사용하는 package 계약을 만든
단계라 별도 benchmark JSON이 없습니다. 번호가 빠진 것이 데이터 누락을 뜻하지
않습니다.

## JSON에서 먼저 볼 필드

- `result_origin`: 실제 local 실행인지, 외부 보고인지
- `environment` 또는 `host`: Python, PyTorch, device와 platform
- `command`: 결과를 만든 명령
- `sources`: 해석에 사용한 논문·구현 ID
- `interpretation` / `limitations`: 이 결과로 주장할 수 없는 범위
- budget·count 필드: step, prompt, token, forward를 공정하게 비교하는 기준

서로 다른 task나 reward의 숫자를 한 표에 놓고 우열로 읽지 마세요. 특히 Agentic
RL의 success/process reward는 TinyReasoning verifier reward와 의미가 다릅니다.

## 운영 감사와의 경계

노트북 실행 manifest, 사용자 여정, CI·Colab 최종 감사처럼 강의 내용을 이해하는
데 필요하지 않은 운영 로그는 공개 학습 문서에서 분리합니다. 정기 검증에서 새로
만든 로그는 GitHub Actions artifact로 보존하며, 이 폴더에는 학습 주장과 직접
연결되는 C1~C9 근거만 둡니다.

출처 분류와 라이선스 판단은 [provenance](../provenance.md), 결과의 한계는
[알려진 한계](../known-limitations.md)를 함께 확인하세요.
