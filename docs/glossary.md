# 용어집

강좌에서 한 개념을 여러 번 다른 말로 부르지 않기 위한 canonical 용어집이다.
영어 mirror에서도 괄호 속 영문·기호를 같은 의미로 사용한다.

| 한국어 | 영문·기호 | 이 저장소에서의 뜻 |
|---|---|---|
| 에이전트 | agent | 관측을 받아 action을 고르는 학습 주체 |
| 환경 | environment | action에 다음 관측, reward, 종료 신호를 돌려주는 시스템 |
| 상태 | state, `s_t` | 미래의 전이를 예측하기에 충분한 환경 정보 |
| 관측 | observation, `o_t` | agent가 실제로 볼 수 있는 정보; state와 같지 않을 수 있음 |
| 행동 | action, `a_t` | classic RL의 선택 또는 LLM의 response token/tool call |
| 정책 | policy, `πθ(a|s)` | 상태/관측에서 행동 확률을 내는 함수 |
| 보상 | reward, `r_t` | 한 전이 직후의 scalar feedback |
| 수익 | return, `G_t` | 현재 이후 할인 reward의 합 |
| 가치 | value, `Vπ(s)` | 정책을 따를 때 기대하는 return |
| 행동가치 | action value, `Qπ(s,a)` | 상태에서 action을 한 뒤 기대하는 return |
| 이점 | advantage, `A(s,a)` | action이 상태의 평균 선택보다 얼마나 나은지 나타내는 값 |
| 할인율 | discount, `γ` | 먼 reward에 주는 상대 가중치 |
| 종료 | terminated | task의 MDP 종료 상태에 도달함; bootstrap하지 않음 |
| 잘림 | truncated | time limit 등 외부 이유로 rollout이 끊김; 경우에 따라 bootstrap함 |
| 궤적 | trajectory | 순서가 있는 observation/action/reward/termination 기록 |
| 온정책 | on-policy | 현재 학습 policy가 모은 데이터로 update |
| 오프정책 | off-policy | 다른/이전 behavior policy의 데이터도 사용 |
| 부트스트랩 | bootstrap | 다음 state의 추정 value로 현재 target을 구성 |
| 리플레이 버퍼 | replay buffer | 과거 transition을 저장하고 다시 sample하는 저장소 |
| 목표 네트워크 | target network | 안정적 target을 위해 늦게 갱신하는 parameter 사본 |
| 정책 경사 | policy gradient | expected return의 policy parameter gradient |
| 기준선 | baseline | 기대 gradient를 바꾸지 않고 variance를 줄이는 비교값 |
| 크리틱 | critic | value/advantage를 추정해 actor를 돕는 모델 |
| 일반화 이점 추정 | GAE | TD residual의 지수 가중합으로 advantage를 만드는 방법 |
| 중요도 비율 | importance ratio | `π_current(a)/π_old(a)`; 다른 policy 분포를 보정 |
| 클리핑 | clipping | 지나치게 큰 policy update의 목적함수 이득을 제한 |
| 엔트로피 | entropy | policy 분포의 불확실성/다양성 |
| KL 발산 | KL divergence | 두 확률분포의 비대칭 차이; 방향을 항상 표시 |
| 지도 미세조정 | SFT | 정답 response의 likelihood를 높이는 supervised fine-tuning |
| 선호 쌍 | preference pair | 같은 prompt에 대한 chosen/rejected response |
| 보상 모델 | reward model, RM | preference에서 response scalar score를 학습하는 모델 |
| 참조 정책 | reference policy | KL 또는 preference objective의 기준이 되는 frozen policy |
| 인간 피드백 강화학습 | RLHF | 사람 preference/reward model을 이용한 RL 정렬 계열 |
| AI 피드백 강화학습 | RLAIF | 사람 대신/함께 AI feedback을 이용하는 RL 정렬 계열 |
| 검증 가능 보상 강화학습 | RLVR | 정답 verifier처럼 자동 검증 가능한 reward를 이용하는 RL |
| 직접 선호 최적화 | DPO | online rollout 없이 preference log-ratio를 최적화하는 objective |
| 그룹 상대 정책 최적화 | GRPO | 같은 prompt completion group의 상대 reward로 advantage를 구성 |
| leave-one-out | RLOO | 각 sample 기준선으로 같은 group의 나머지 reward 평균을 사용 |
| Dr. GRPO | Dr. GRPO | GRPO의 sample/length-level 편향을 분석하고 reduction을 보정한 변형 |
| DAPO | DAPO | Clip-Higher, Dynamic Sampling, token loss, overlong shaping 조합 |
| 그룹 시퀀스 정책 최적화 | GSPO | sequence-level importance ratio와 clipping을 쓰는 정책 최적화 |
| 행동 마스크 | action mask | policy loss에 포함할 model action token 위치 |
| attention 마스크 | attention mask | model attention에서 실제 token과 padding을 구분 |
| 과정 보상 | process reward | 중간 reasoning/tool step의 품질에 주는 reward |
| 결과 보상 | outcome reward | 최종 task 성공/실패에 주는 reward |
| credit assignment | credit assignment | 결과를 어느 step/action/token의 update에 귀속할지 정하는 과정 |
| 실험 카드 | experiment card | config, 환경, provenance, 비용, 결과와 한계를 묶은 실행 기록 |
| clean-room 재구현 | clean-room reimplementation | upstream 코드를 복사·변형하지 않고 공개 수식·명세로 새로 작성 |
| 외부 수동 검증 | external-manual | 필요한 hardware/service가 없어 명령과 기대 계약만 검증된 상태 |

## 혼동하지 않는 표기

- `old`는 rollout을 만든 고정 policy snapshot, `current`는 update 중인 policy,
  `reference`는 KL/preference 기준 policy다.
- `reward`는 즉시 feedback, `return`은 누적 reward, `advantage`는 baseline 대비
  상대적 좋음이다.
- `prompt mask`, `response/action mask`, `attention mask`, `tool/environment mask`를
  모두 “mask” 한 단어로 축약하지 않는다.
- forward KL은 `KL(reference || policy)`처럼 방향을 쓰고, reverse KL도 마찬가지다.
- 논문 수치(`paper_reported`), upstream 수치(`upstream_reported`), 이 저장소에서
  측정한 수치(`local_executed`)를 섞지 않는다.
