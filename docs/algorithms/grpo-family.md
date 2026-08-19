# GRPO 계열: 이름보다 식을 먼저 보기

> 구현 상태: C7 toy clean-room 구현. 이 문서는 `deepseekmath-grpo-2024`,
> `rloo-2024`, `dr-grpo-2025`, `gspo-2025`를 서로 다른 알고리즘으로 구분한다.
> 짧은 CPU 실험은 수식과 코드의 동작 확인용이며 논문 규모의 성능 재현이 아니다.

## 1. 공통 출발점: 한 prompt에서 여러 답 뽑기

prompt `q`마다 `G`개 completion을 뽑고 verifier reward `r_i`를 얻는다. GRPO는
별도 critic 대신 같은 group의 평균과 표준편차로 advantage를 만든다.

\[
A_i = \frac{r_i-\operatorname{mean}(r_1,\ldots,r_G)}
{\operatorname{std}(r_1,\ldots,r_G)+\epsilon}
\]

이 구현은 PyTorch의 population standard deviation(`unbiased=False`)을 사용한다.
논문 식의 group 통계에 직접 대응하고 작은 group에서 정의가 단순하기 때문이다.
sample standard deviation도 가능한 선택이지만 `G`가 작을 때 scale이 달라지므로
실험 카드 없이 바꾸면 안 된다.

모든 reward가 같으면 표준편차가 0이다. 작은 수로 나눠 noise를 키우지 않고 그
group의 advantage를 정확히 0으로 만든다. group size 1은 상대 비교가 불가능하므로
명시적으로 거부한다. 동일 completion이나 동일 reward가 중복돼도 NaN을 만들지
않는다. 이 계약은 `group_relative_advantages`와 analytic test에 고정돼 있다.

## 2. DeepSeekMath식 GRPO

old policy로 뽑은 각 response token에 대해

\[
\rho_t=\exp(\log\pi_\theta(a_t|s_t)-\log\pi_{old}(a_t|s_t))
\]

를 만들고 PPO식 clipped surrogate를 사용한다. 한 sequence의 `A_i`가 그
response의 모든 action token에 broadcast된다. 기본 reduction은 **각 response의
token 평균을 먼저 구한 뒤 sequence 평균**이다.

reference KL에는 sampled log-ratio 자체가 아니라 DeepSeekMath의 pointwise `k3`
estimator를 쓴다.

\[
K_t=\exp(\log\pi_{ref}-\log\pi_\theta)
-(\log\pi_{ref}-\log\pi_\theta)-1
\]

코드는 cancellation을 줄이기 위해 `expm1`을 사용한다. 동일 weight를 복제한 두
Transformer도 CPU kernel에서 약 `1e-6` log-prob 차이가 날 수 있다. 이 차이가
가짜 KL gradient가 되지 않도록 `1e-5` 이하는 값과 gradient를 0으로 만드는
수치 guard를 두고 별도 test로 검증한다. 이는 목적함수 변경이 아니라 수학적으로
같아야 할 clone-equality를 보존하는 구현 선택이다.

| 개념 | 구현 |
|---|---|
| group 평균/std advantage | `group_relative_advantages` |
| token ratio·clip·reference KL | `grpo_loss` |
| sequence/token/fixed reduction | `reduce_group_tokens` |
| group rollout과 frozen old/reference | `train_group_policy` |

critic을 제거하면 value model의 메모리와 학습 오류가 사라진다. 대신 prompt마다
`G`개 답을 생성해야 하고, group reward가 전부 같으면 학습 신호가 0이다. 즉
“critic-free”는 “공짜”가 아니라 rollout 비용과 within-group variance에 비용을
옮긴 선택이다.

## 3. 이름이 다른 네 변형

| variant | baseline/advantage | ratio와 update | 길이 reduction |
|---|---|---|---|
| GRPO | group mean/std | token ratio + PPO clip + k3 KL | sequence별 token 평균 |
| RLOO | 나머지 `G-1` reward 평균 | 전체 response를 한 action으로 보는 REINFORCE, no clip | sequence log-prob 합 |
| Dr. GRPO | group mean만 빼고 std로 나누지 않음 | token ratio + clip | 전체 합을 `batch × max_response_length`로 나눔 |
| GSPO | group mean/std | response token log-ratio의 평균을 exp한 sequence ratio + sequence clip | sequence objective |

RLOO advantage는

\[
A_i=r_i-\frac{1}{G-1}\sum_{j\ne i}r_j
\]

이고 completion 전체의 log-prob 합에 한 번 곱한다. 논문 경로는 on-policy
REINFORCE이므로 이 repo는 RLOO의 `update_epochs=1`을 강제한다.

Dr. GRPO는 reward 표준편차 normalization과 response별 길이 normalization이
만드는 sample/length bias를 함께 제거한다. 고정 최대 길이 분모는 짧은 response의
각 token이 더 큰 가중치를 받는 현상을 없애지만, 실제 token 수가 적은 batch에서는
gradient scale도 작아진다.

GSPO의 sequence ratio는

\[
s_i=\exp\left(\frac{1}{|o_i|}\sum_t
(\log\pi_\theta-\log\pi_{old})\right)
\]

이다. token ratio를 평균한 것이 아니라 **log-ratio 평균의 지수**, 즉 기하평균
ratio다. 논문 설정을 드러내기 위해 toy preset도 매우 좁은 asymmetric clip
`3e-4/4e-4`를 별도 이름으로 보존한다.

## 4. length ablation을 읽는 법

두 response의 token objective 합이 각각 2와 3이고 길이가 2와 1이면:

- sequence mean: `(2/2 + 3/1) / 2 = 2`
- global token mean: `(2+3)/(2+1) = 5/3`
- Dr. GRPO, max length 2: `(2+3)/(2×2) = 1.25`

세 값은 모두 “평균”처럼 보이지만 학습 가중치가 다르다. C7 analytic test와
component report는 이 차이를 숫자로 고정한다.

## 5. 실행과 해석 제한

```bash
rl-study train --config configs/toy/grpo.yaml --stop-after 1 --json
rl-study train --config configs/toy/rloo.yaml --json
python scripts/run_group_comparison.py --steps 3 --prompt-batch-size 1 --group-size 4
```

C7 local run에서는 10개 variant가 동일 SFT parameter hash에서 시작했다. non-dynamic
variant는 prompt 순서와 rollout token 예산도 같았다. 단 3 rollout step·seed 42의
exact match로 알고리즘 순위를 만들 수 없으며, 결과의 주 목적은 variant와 비용이
실제로 분리됐는지 확인하는 것이다.

## Sources

- `deepseekmath-grpo-2024`: group-relative advantage, token ratio, k3 KL
- `rloo-2024`: sequence-action REINFORCE와 leave-one-out baseline
- `dr-grpo-2025`: reward-std/length normalization bias와 고정 분모
- `gspo-2025`: sequence-level importance ratio와 clipping
- `repo-deepseek-math`, `repo-understand-r1-zero`, `framework-trl`,
  `framework-verl`: variant 이름과 현대 framework 차이의 교차검산
