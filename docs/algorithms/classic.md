# 고전 RL: 식에서 코드까지

> source: `sutton-barto-rl2`, `dqn-2013`, `gae-2015`, `ppo-2017`,
> `repo-spinningup`
> provenance: 논문·책 수식을 바탕으로 한 clean-room PyTorch 구현

이 문서는 L03~L08이 사용하는 canonical implementation note다. 모든 shape은
batch를 생략하지 않고 쓰며, 논문 수치와 이 저장소의 4×4 GridWorld 결과를
구분한다.

## 1. 공통 환경과 종료 의미

`TinyGridWorld`는 state 16개, action 4개, goal reward 1, 그 외 step reward
-0.01, 최대 32 step이다.

```text
terminated=True : MDP goal에 도달 → TD target에서 bootstrap 0
truncated=True  : time limit로 잘림 → next value로 bootstrap 가능, 새 episode로
                  GAE recurrence가 넘어가지는 않음
```

이 구분은 [`td_target`](https://github.com/BangProx/RL-study/blob/main/src/rl_study/algorithms/tabular.py),
[`dqn_loss`](https://github.com/BangProx/RL-study/blob/main/src/rl_study/algorithms/dqn.py),
[`generalized_advantage_estimate`](https://github.com/BangProx/RL-study/blob/main/src/rl_study/math/returns.py)에서 같은
의미를 가진다. 두 signal을 단순 `done`으로 합치면 time-limit state의 value를
체계적으로 낮게 추정할 수 있다.

관련 test: `test_td_and_q_targets_disable_only_terminal_bootstrap`,
`test_dqn_terminal_removes_bootstrap_but_truncation_keeps_it`,
`test_truncation_bootstraps_but_stops_gae_recurrence`.

## 2. Dynamic Programming

Bellman optimality backup:

$$
V_{k+1}(s)=\max_a\sum_{s'}p(s'|s,a)
\left[r(s,a,s')+\gamma\mathbf{1}_{s'\text{ nonterminal}}V_k(s')\right].
$$

| 수식 | tensor/code |
|---|---|
| $p(s'|s,a)$ | `transition_probabilities: [S,A,S]` |
| $r(s,a,s')$ | `rewards: [S,A,S]` |
| terminal indicator | `terminal_states: bool [S]` |
| sum over $s'$ | `.sum(dim=-1)` |
| max over $a$ | `.max(dim=-1)` |

[`value_iteration`](https://github.com/BangProx/RL-study/blob/main/src/rl_study/algorithms/tabular.py)은 synchronous backup을
쓴다. in-place asynchronous update도 더 빨리 수렴할 수 있지만, 한 iteration의
수식과 tensor를 그대로 보여주고 policy iteration과 공정하게 비교하기 위해
synchronous 방식을 선택했다.

[`policy_iteration`](https://github.com/BangProx/RL-study/blob/main/src/rl_study/algorithms/tabular.py)은 현재 deterministic
policy를 수렴할 때까지 평가한 뒤 greedy improvement를 한다. 두 방법은 이
환경에서 같은 value에 도달하며 `test_value_and_policy_iteration_agree`가 검증한다.

흔한 오류:

- goal value에서 다시 reward를 무한히 bootstrap한다.
- transition row 합이 1인지 확인하지 않는다.
- `argmax` action dimension과 next-state sum dimension을 바꾼다.

## 3. Monte Carlo와 TD(0)

Monte Carlo reward-to-go:

$$G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots.$$

[`monte_carlo_returns`](https://github.com/BangProx/RL-study/blob/main/src/rl_study/algorithms/tabular.py)은 뒤에서 앞으로
`running = reward[t] + gamma * running`을 계산한다. first-visit update는 한
episode에서 같은 state가 다시 나와도 처음의 return만 incremental mean에 넣는다.
`train_mc_prediction`은 value와 visit count를 함께 받아 episode 경계 resume를
지원한다.

TD(0) target:

$$y_t=r_t+\gamma(1-d_t)V(s_{t+1}),\qquad
V(s_t)\leftarrow V(s_t)+\alpha(y_t-V(s_t)).$$

여기서 $d_t$는 **terminated**다. MC는 episode 전체를 본 뒤 unbiased return을
얻는 대신 variance가 크고, TD는 bootstrap bias를 받아들이는 대신 매 step
update할 수 있다. 이 bias/variance trade-off를 “어느 쪽이 항상 낫다”로
요약하지 않는다. `train_td0_prediction`도 이전 value와 episode offset을 받아
연속 실행과 재개 실행이 같은 update 순서를 갖는다.

관련 code/test:

- `monte_carlo_returns`, `mc_value_update`, `td_target`, `td0_update`
- `test_mc_first_visit_update_and_returns`, `test_discounted_returns_hand_calculation`

## 4. Q-learning

off-policy target:

$$y_t=r_t+\gamma(1-d_t)\max_{a'}Q(s_{t+1},a').$$

behavior는 epsilon-greedy지만 target은 greedy다. 이 둘을 한 policy라고 부르면
on/off-policy 차이를 잃는다.

[`q_learning_target`](https://github.com/BangProx/RL-study/blob/main/src/rl_study/algorithms/tabular.py)은 `[B,A]`의 마지막
dimension을 max하고 [`train_q_learning`](https://github.com/BangProx/RL-study/blob/main/src/rl_study/algorithms/tabular.py)은
현재 state/action 한 원소만 update한다. episode별 exploration RNG는
`seed + episode_id * 1_000_003`으로 독립시켜 checkpoint resume 후에도 같은
trajectory를 만든다.

대안:

- SARSA: behavior가 실제 고른 next action을 target에 사용한다.
- Expected SARSA: behavior distribution의 기대 Q를 사용한다.
- Double Q-learning: 같은 noisy estimate로 선택·평가하며 생기는 max bias를 줄인다.

승인 범위에서는 Q-learning만 구현하고 나머지는 대안으로 설명한다.

## 5. DQN

$$
y_t=r_t+\gamma(1-d_t)\max_{a'}Q_{\theta^-}(s_{t+1},a'),\qquad
L=\operatorname{Huber}(Q_\theta(s_t,a_t), y_t).
$$

| 역할 | 구현 | gradient |
|---|---|---|
| online $Q_\theta$ | `DQNNetwork policy` | 있음 |
| target $Q_{\theta^-}$ | `DQNNetwork target` | `requires_grad=False` + `no_grad` |
| replay | `ReplayBuffer` | tensor로 sample한 뒤 없음 |
| hard sync | `hard_update` | parameter copy, optimizer 밖 |

MSE 대신 Huber loss를 쓰면 큰 초기 TD error의 gradient를 제한한다. MSE도 가능한
대안이며 더 직접적이지만 outlier에 민감하다. target은 `detach()` 한 tensor만
만드는 데 그치지 않고 network parameter 자체도 frozen 상태로 유지한다.

Double DQN toggle은 online network로 action을 선택하고 target network로 그
action을 평가한다. 기본 lesson은 원 DQN의 max target을 사용한다.

### 실제 instability ablation

2026-08-19 M4 CPU, seed 42, 250 episode에서 hard target sync(40 env step)는
validation success 1.0, frozen initial target은 0.0이었다. 이는 target network가
모든 문제에서 정확히 이 차이를 만든다는 주장이 아니라, 이 작은 seed에서
움직이지 않는 target이 학습을 막는 failure demonstration이다. exact record는
[`C4_CLASSIC_BENCHMARK.json`](../research/C4_CLASSIC_BENCHMARK.json)에 있다.

관련 test: `test_dqn_target_network_is_detached`, lifecycle resume parity.

## 6. REINFORCE

$$L_{PG}=-\frac1T\sum_t\log\pi_\theta(a_t|s_t)(G_t-b_t).$$

[`reinforce_loss`](https://github.com/BangProx/RL-study/blob/main/src/rl_study/algorithms/policy_gradient.py)은 advantage를
detach한다. reward가 양수일 때 gradient descent가 selected action log-prob을
올려야 하므로 맨 앞의 음수가 필요하다. `test_reinforce_sign_increases...`가 이
부호를 검산한다.

기본 trainer의 baseline은 **이전 episode까지의** return으로 만든 exponential
running mean이다. 현재 episode 표본 평균을 같은 표본의 baseline으로 사용하면
직관은 쉽지만 작은 batch에서 편향 논의가 생기므로 피했다. baseline은 expected
gradient를 바꾸지 않는 action-independent 값이어야 한다.

baseline on/off 모두 이 toy seed에서는 success 1.0이었다. 이 한 번의 최종
success만으로 variance 감소를 주장하지 않으며 notebook에서는 여러 seed의
gradient variance를 직접 비교한다.

## 7. Actor-Critic

$$
\delta_t=r_t+\gamma(1-d_t)V_\phi(s_{t+1})-V_\phi(s_t),
$$

$$
L_{actor}=-\log\pi_\theta(a_t|s_t)\,\operatorname{stopgrad}(\delta_t),
\qquad L_{value}=\tfrac12\delta_t^2.
$$

[`actor_critic_loss`](https://github.com/BangProx/RL-study/blob/main/src/rl_study/algorithms/policy_gradient.py)은 actor loss의
advantage를 끊지만 value loss에서는 critic gradient를 보존한다. next value는
TD target 쪽이므로 detach한다. actor와 critic을 완전히 별도 optimizer로 둘 수도
있다. 작은 모델에서는 한 optimizer가 간단하지만, loss coefficient와 learning
rate 간 coupling이 생기는 trade-off가 있다.

## 8. GAE

$$
\hat A_t^{GAE(\gamma,\lambda)}=
\delta_t+(\gamma\lambda)\delta_{t+1}+\cdots.
$$

[`generalized_advantage_estimate`](https://github.com/BangProx/RL-study/blob/main/src/rl_study/math/returns.py)의 input:

- `rewards: [T,...]`
- `values: [T+1,...]`
- `terminated`, `truncated: bool [T,...]`

`lambda=0`은 one-step TD residual, `lambda=1`은 bootstrap 경계를 포함한 긴
return 쪽으로 간다. truncation step의 delta에는 next value를 쓰지만 그 뒤 새
episode의 delta를 앞 episode로 전달하지 않는다.

analytic test는 $delta=[0.86,0.6]$, $\gamma=0.9$, $\lambda=0.8$에서
$A=[1.292,0.6]$을 정확히 확인한다.

## 9. PPO

$$
r_t(\theta)=\exp(\log\pi_\theta(a_t|s_t)-
\log\pi_{\theta_{old}}(a_t|s_t)),
$$

$$
L^{clip}=-\mathbb{E}_t\left[
\min(r_t\hat A_t,\operatorname{clip}(r_t,1-\epsilon_l,1+\epsilon_h)\hat A_t)
\right].
$$

[`ppo_policy_loss`](https://github.com/BangProx/RL-study/blob/main/src/rl_study/algorithms/ppo.py)은 current log-prob에만
gradient를 흘리고 old log-prob과 advantage를 detach한다. 반환값은 scalar loss
외에도 ratio, unclipped/clipped objective, approximate KL, clip fraction이다.

### positive/negative advantage에서 clip이 다른 이유

- $A>0$: ratio가 너무 커져 얻는 이득을 upper clip이 막는다.
- $A<0$: ratio가 너무 작아져 나쁜 action을 과도하게 억제하는 이득을 lower
  clip이 막는다.

ratio=1과 두 부호의 경계는 `test_ppo_ratio_one...`,
`test_ppo_clip_handles_positive_and_negative_advantages`가 손계산한다.

value clipping, KL early stop, separate minibatch는 가능한 옵션이다. classic toy
기본은 unclipped MSE value loss와 full-episode batch를 써서 식을 짧게 유지한다.
LLM RLHF-PPO에서는 token mask·reference KL·sequence reward 때문에 같은 PPO
이름이어도 reduction이 달라지며 L10/C6에서 별도 module로 구현한다.

## 10. Checkpoint/resume 공정성

[`classic_runner`](https://github.com/BangProx/RL-study/blob/main/src/rl_study/training/classic_runner.py)은 Q table, online/
target DQN, replay contents, optimizer, running baseline, episode/environment cursor와
RNG를 저장한다. schedule의 분모는 최종 configured steps로 고정한다.

`tests/integration/test_classic_runner_lifecycle.py`는 다섯 학습 agent 각각에 대해:

```text
20 episodes continuous
== 10 episodes → atomic checkpoint → resume → 10 episodes
```

의 모든 model/buffer tensor가 `rtol=0, atol=0`인지 확인한 뒤 checkpoint에서
evaluation을 다시 실행한다. `training.steps` 연장은 허용하지만 model/data/
algorithm 의미가 바뀌면 resume immutable hash가 load 전에 거부한다.

## 11. 이 결과를 해석하는 법

local 기본 run은 Q-learning, DQN, REINFORCE, actor-critic, PPO가 모두 success
1.0에 도달했다. 이 결과는 다음을 의미하지 않는다.

- PPO가 다른 task에서도 항상 더 낫다.
- 50 deterministic evaluation episode가 uncertainty를 완전히 측정한다.
- Atari DQN이나 논문 PPO 결과를 재현했다.

의미하는 것은 implementation이 finite update를 만들고, 이 저장소의 작은 MDP를
학습하며, 중단/재개가 연속 실행과 정확히 같다는 sanity evidence다.
