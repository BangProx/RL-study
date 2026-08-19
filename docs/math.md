# 수학 최소 도구함

증명 전체보다 **수식의 기호가 코드에서 어떤 tensor인가**에 초점을 맞춥니다.
shape의 첫 축은 보통 batch `B`, 두 번째는 time/token `T`입니다.

## 확률, 기대값, 분산

이산 행동의 확률은 $p(a) \ge 0$, $\sum_a p(a)=1$입니다.

$$
\mathbb{E}[X]=\sum_x p(x)x,\qquad
\mathrm{Var}(X)=\mathbb{E}[(X-\mathbb{E}X)^2].
$$

sample 평균은 기대값의 추정치이며, baseline과 더 많은 rollout은 gradient의
분산을 낮추는 데 쓰입니다. `torch.multinomial`은 확률에 따른 표본을,
`argmax`는 가장 큰 값 하나를 선택합니다.

## log-probability, entropy, cross-entropy

긴 sequence 확률의 곱은 매우 작아지므로 합으로 바꿉니다.

$$
\log \pi_\theta(y\mid x)=\sum_{t\in\text{action}}
\log \pi_\theta(y_t\mid x,y_{<t}).
$$

prompt와 padding은 합에서 제외합니다. 구현의 `[B,T]` log-prob에 bool
`action_mask`를 곱한 뒤, 문서에 적힌 reduction으로 줄입니다.

$$
H(\pi)=-\sum_a \pi(a)\log\pi(a),\qquad
H(q,p)=-\sum_a q(a)\log p(a).
$$

entropy가 낮아지면 policy가 한 행동에 몰렸다는 신호일 수 있지만, 무조건 나쁜
것은 아닙니다. task 성능, KL, 길이와 함께 봅니다.

## KL divergence: 방향을 생략하지 않기

$$
D_{KL}(p\Vert q)=\mathbb{E}_{a\sim p}
\left[\log\frac{p(a)}{q(a)}\right] \ge 0.
$$

$D_{KL}(p\Vert q) \ne D_{KL}(q\Vert p)$입니다. RLHF의 sampled token KL은
`log π_policy - log π_reference`를 policy rollout에서 평균한 추정치라 finite
sample에서 음수가 나올 수 있습니다. 진짜 분포 KL이 음수라는 뜻이 아닙니다.

## gradient와 detach

gradient descent는 $\theta\leftarrow\theta-\alpha\nabla_\theta L$입니다.
PyTorch에서는 `loss.backward()`가 graph를 따라 `.grad`를 채웁니다.

- current policy: gradient가 흘러야 함
- advantage/return target: 보통 `detach`
- old policy, reference policy, reward model: policy update 동안 frozen
- value model: value loss로만 update

이 소유권을 테스트하지 않으면 loss 값은 맞아도 엉뚱한 model이 학습될 수 있습니다.

## return, Bellman, TD

$$
G_t=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k},\qquad
V^\pi(s)=\mathbb{E}_\pi[r+\gamma V^\pi(s')].
$$

TD target은 $r+\gamma V(s')$이고 terminal이면 다음 value를 0으로 둡니다.
time-limit `truncated`는 task terminal과 다르므로 bootstrap 규칙을 명시해야 합니다.

## policy gradient와 advantage

$$
\nabla_\theta J(\theta)=
\mathbb{E}[\nabla_\theta\log\pi_\theta(a_t\mid s_t)A_t].
$$

최대화식을 최소화 loss로 구현하면 앞에 음수가 붙습니다.
`-(log_prob * advantage.detach()).mean()`의 부호가 핵심입니다.

GAE는 TD residual $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$를 섞습니다.

$$
\hat A_t^{GAE(\gamma,\lambda)}=
\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}.
$$

$\lambda=0$은 one-step TD 쪽, $\lambda=1$은 Monte Carlo 쪽입니다.

## PPO

$$
r_t(\theta)=\exp(\log\pi_\theta-\log\pi_{old}),
$$

$$
L^{clip}=\mathbb{E}\left[
\min(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)
\right].
$$

`old_log_probs.detach()`가 빠지면 old policy까지 움직입니다. 양수·음수 advantage에서
`min`과 clipping의 동작이 다르므로 hand-calculation test가 필요합니다.

## DPO

chosen/rejected response의 policy-reference log-ratio 차이를 $z$라 하면:

$$
L_{DPO}=-\log\sigma\left(\beta[
(\log\pi_\theta(y_w|x)-\log\pi_{ref}(y_w|x))-
(\log\pi_\theta(y_l|x)-\log\pi_{ref}(y_l|x))]
\right).
$$

DPO는 고정 preference data의 offline objective입니다. online rollout RL과 같다고
부르지 않습니다.

## GRPO와 DAPO

GRPO는 한 prompt의 group reward를 평균·표준편차로 정규화해 advantage를 만듭니다.

$$
\hat A_i=\frac{r_i-\operatorname{mean}(r_{1:G})}
{\operatorname{std}(r_{1:G})+\epsilon}.
$$

모든 reward가 같으면 학습 신호가 0입니다. DAPO는 비대칭 Clip-Higher, informative
group을 찾는 bounded dynamic sampling, global token reduction, soft overlong
shaping을 결합합니다. 각 구성요소는 [DAPO 구현 노트](algorithms/dapo.md)에서
독립적으로 끄고 검사합니다.

## reduction과 mask 체크리스트

1. loss tensor의 shape를 적었는가?
2. prompt, response, EOS, padding, tool output mask를 구분했는가?
3. sequence 평균과 global token 평균 중 무엇인지 적었는가?
4. gradient가 흐를 model과 끊길 model을 테스트했는가?
5. 빈 mask, zero-variance group, NaN/Inf를 거부하는가?

Sources: `sutton-barto-rl2`, `gae-2015`, `ppo-2017`, `dpo-2023`,
`deepseekmath-grpo-2024`, `dapo-2025`. 전체 metadata는
[출처 원장](sources.yml)에 있습니다.
