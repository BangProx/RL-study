# Agentic RL: 답 하나가 아니라 행동의 연쇄를 학습하기

## 30초 지도

일반 reasoning RL은 흔히 `prompt → 긴 response → reward`를 한 action으로 본다.
Agentic RL은 중간에 도구와 환경이 끼어든다.

```text
observation → model action → parser → allowlisted tool → observation → ... → final
```

따라서 “어떤 문자열을 생성했는가” 외에도 **어느 step의 행동인지, 도구가 무엇을
돌려줬는지, episode가 끝났는지, 마지막 성공을 앞선 행동에 어떻게 돌려줄지**를
보존해야 한다. 이 저장 단위가 `AgentStep`과 `AgentTrajectory`다.

## single-response MDP와 step-level MDP

| 관점 | action | reward 위치 | 장점 | 놓치기 쉬운 것 |
|---|---|---|---|---|
| single response | 도구 호출까지 포함한 전체 문자열 | 응답 끝 | 구현이 단순함 | 실제 환경 전이와 중간 실패 |
| step-level | 각 model action/tool call | 매 step + 종료 | 도구 feedback과 credit 관찰 | stale rollout, 긴 context, 더 큰 비용 |

이 저장소는 Agent-R1의 step-level MDP 관점과 Agent Lightning의 agent 실행/학습
분리 아이디어를 참고하되 코드는 clean-room으로 작게 구현한다. 외부 repository의
코드는 복사하지 않았다. 출처 ID는 `agent-r1-2025`, `agent-lightning-2025`,
`repo-agent-r1`, `framework-agent-lightning`이다.

## 두 개의 완전 오프라인 환경

### `CalculatorToolEnv`

- action: `CALL calculator {"expression":"2+3"}` 또는 `FINAL 5`
- parser는 정확히 한 개의 JSON string 인자만 허용한다.
- 계산은 Python `eval`이 아니라 숫자와 `+ - * /`만 방문하는 AST allowlist다.
- 이름 접근, 함수 호출, shell, 파일, network는 실행 경로 자체가 없다.

### `LocalLookupEnv`

- action: `CALL lookup {"key":"kr"}` 또는 `FINAL Seoul [kr]`
- immutable한 네 항목짜리 메모리 corpus만 읽는다.
- 정답과 `[key]` citation이 모두 맞아야 성공이다.
- 존재하지 않는 key와 반복 호출은 별도 penalty를 받는다.

두 환경 모두 `reset/step`, visible tool schema, invalid action, tool timeout,
`terminated`와 `truncated`를 typed 값으로 돌려준다. outcome reward는 종료 step에만
존재한다. `AgentStep.reward = process_reward + outcome_reward` 한 곳에서만 더하므로
double counting하지 않는다.

## 모델은 정확히 무엇을 학습하는가

Toy 모델은 open-vocabulary 생성 대신 유한한 admissible action 후보를 LM으로
채점한다. 이것은 작은 CPU 실습을 안정적으로 만드는 scaffold이며 논문의 대규모
agent decoder와 같다고 주장하지 않는다.

후보 (a)의 action-token 평균 log likelihood를 (s_\theta(a, h_t))라 하면:

\[
s_\theta(a,h_t)=\frac{1}{|a|}\sum_{i\in a}
\log p_\theta(a_i\mid h_t,a_{<i}),\qquad
\pi_\theta(a\mid h_t)=\operatorname{softmax}_{a\in\mathcal A_t}s_\theta(a,h_t).
\]

평균을 쓰는 이유는 긴 action이 log probability 합만으로 자동 불리해지는 길이
편향을 줄이기 위해서다. 다른 선택지는 합, 길이 penalty, 실제 autoregressive
sampling이다. 합은 원 LM likelihood에 가장 충실하지만 후보 길이 차이에 민감하고,
open-vocabulary sampling은 더 현실적이나 이 강의의 CPU budget과 parser 성공률을
크게 악화시킨다.

중요한 mask 경계는 다음과 같다.

| token | attention | 현재 policy loss |
|---|---:|---:|
| 현재 observation/prompt | O | X |
| 과거 model action | O, 다음 step의 context | X |
| tool/environment output | O, 다음 step의 context | X |
| 현재 model action과 EOS | O | O |
| padding | X | X |

rollout 때 생성된 `action_token_ids`를 그대로 저장한다. update 직전에 action text를
다시 tokenize한 결과가 다르면 `RetokenizationDriftError`로 중단한다. 편리하다는
이유로 `Token → text → Token` 결과를 원본이라고 간주하지 않는다.

## 두 credit assignment 비교

### Baseline: final outcome broadcast

\[
c_t = R_{\text{outcome}},\qquad
L=-\frac{1}{T}\sum_t c_t\log\pi_\theta(a_t\mid h_t).
\]

마지막 outcome을 모든 step에 복사한다. 구현은 간단하지만 timeout으로 outcome이
0이면 올바른 중간 tool call도 학습 신호를 전혀 받지 못한다.

### Step reward를 포함한 discounted return

\[
G_t = p_t + o_t + \gamma G_{t+1},\qquad
L=-\frac{1}{T}\sum_t G_t\log\pi_\theta(a_t\mid h_t).
\]

여기서 (p_t)는 process reward, (o_t)는 종료 때 한 번만 주는 outcome reward다.
현재 toy 구현에는 value model이 없으므로 GAE가 아니라 Monte Carlo return을 쓴다.
GAE를 쓰려면 state value, bootstrap 규칙, value loss와 별도 optimizer가 필요하다.

## 실패 실험: 반드시 실패해야 안전하다

| failure | 이 저장소의 재현/차단 |
|---|---|
| context truncation | 최대 길이를 넘으면 조용히 자르지 않고 `ValueError` |
| stale/asynchronous rollout | policy version lag를 검사해 기본 lag 0 외 거부 |
| tool nondeterminism | 같은 입력 반복 비교 helper와 deterministic seed test |
| retokenization drift | rollout 원 token ID와 재인코딩 ID 불일치 시 중단 |
| reward hacking | 호출마다 +0.1인 취약 reward와 반복 호출 penalty를 나란히 test |
| arbitrary command | strict tool allowlist; shell/network API가 toy env에 없음 |

stale 검사는 async rollout 자체를 금지한다는 뜻이 아니다. worker를 추가한다면
허용할 최대 policy lag와 off-policy correction을 먼저 결정해야 한다는 뜻이다.

## 실행

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 rl-study train \
  --config configs/toy/agentic_broadcast.yaml
TORCH_DEVICE_BACKEND_AUTOLOAD=0 rl-study train \
  --config configs/toy/agentic_returns.yaml
rl-study eval --checkpoint <checkpoint-directory>
```

seed 42, 24 update의 실제 C9 실행에서는 두 방식 모두 validation success 0이었다.
특정 알고리즘이 이겨야 한다는 결론을 만들지 않는다. return 방식의 validation
useful-tool-step rate는 0.1818, broadcast는 0.125였지만 task 8개·seed 1개이므로
우월성 증거가 아니다. 핵심 관찰은 broadcast의 마지막 loss가 `-0.0`이어서 outcome
0 trajectory가 신호를 잃는 반면, return은 process reward 때문에 유한한 non-zero
update를 만들었다는 점이다. 원 실행 수치는 `C9_AGENTIC_BENCHMARK.json`에 있다.

## 선택형 ALFWorld adapter

`AlfWorldAdapter`는 ALFWorld 0.4.2의 이미 설치된 single text environment를 주입받는
경계만 제공한다. 최신 observation의 `admissible_commands`에 없는 명령은 backend를
호출하기 전에 거부한다. 현재 Mac arm64에는 package/assets가 없으므로 상태는
`external-manual`이다. ALFWorld/TextWorld 코드는 vendor하지 않았고, 선택 planner인
Fast Downward의 GPL-3.0 경계도 core와 분리했다. 출처 ID는 `benchmark-alfworld`다.

## 코드 지도

- 계약: `src/rl_study/agentic/types.py`
- parser/환경: `agentic/parser.py`, `agentic/envs.py`
- action-token policy/credit/update: `agentic/policy.py`, `credit.py`, `trajectory.py`
- lifecycle: `training/agentic_runner.py`
- 선택 adapter: `adapters/alfworld.py`
- 검증: `tests/unit/test_agentic_*`, `tests/unit/test_alfworld_adapter.py`,
  `tests/integration/test_agentic_runner.py`
