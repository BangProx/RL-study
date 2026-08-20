# DAPO: 네 부품을 하나씩 켜 보기

> 구현 상태: C7 paper-only clean-room 구현. 감사한 DAPO 공식 repository commit에는
> license가 없었으므로 코드·설정·문서 문장·asset을 복사하거나 변형하지 않았다.
> 아래 구현 근거는 `dapo-2025` 논문과 독립 라이선스 framework 감사 기록이다.

## 1. DAPO는 하나의 새 loss만 뜻하지 않는다

DAPO 논문이 함께 사용한 네 요소를 독립 toggle로 나눴다.

| 요소 | 문제 | 이 repo의 구현 지점 |
|---|---|---|
| Clip-Higher | 좋은 token의 확률 증가가 대칭 upper clip에 빨리 막힘 | `dapo_loss(use_clip_higher=True)` |
| Dynamic Sampling | group이 전부 정답/오답이면 advantage가 0 | `dynamic_sampling_filter` |
| Token-level PG loss | response별 평균이 길이에 주는 가중치 차이 | `dapo_loss(use_token_level_loss=True)` |
| Soft Overlong Punishment | 최대 길이에 걸린 잘린 답의 noisy reward | `overlong_reward_shaping` |

`configs/toy/dapo.yaml`은 네 요소를 모두 켜고, 비교 runner는 none/각 하나/all의
여섯 DAPO variant를 실행한다. 따라서 “DAPO”라는 label만 비교하지 않고 어떤
구성요소가 실제로 달랐는지 experiment card에 남는다.

## 2. Clip-Higher

대칭 PPO clip `[1-ε, 1+ε]` 대신 `[1-ε_low, 1+ε_high]`를 사용한다. 논문의
대표값대로 toy preset은 `ε_low=0.2`, `ε_high=0.28`이다. positive advantage에서
ratio가 1.2를 넘을 때 추가 상승 여유를 주지만, negative advantage의 lower-side
동작은 바꾸지 않는다. 손계산 test가 두 부호를 따로 확인한다.

한 번의 on-policy epoch에서는 처음 ratio가 1이므로 Clip-Higher와 대칭 clip의
차이가 나타나지 않을 수 있다. C7의 3-step component report에서 두 policy hash가
같은 것은 이 조건의 예상 결과이며, “효과 없음”이라는 일반 결론이 아니다.

## 3. Dynamic Sampling과 유한 예산

각 prompt의 group reward가 모두 같으면 버리고 informative group만 학습한다.
논문은 target batch가 찰 때까지 계속 sampling하지만, laptop 실습이 무한히 돌지
않도록 후보 예산을 `target × dynamic_sampling_multiplier`로 제한한다.

```text
candidate group 생성
  ├─ max(reward) == min(reward): rejected
  └─ 서로 다른 reward 존재: selected
target 수 미달 + 예산 소진 → exhausted=true, 부족분을 조용히 대체하지 않음
```

이것은 논문 대비 의도적인 규모 축소다. C7 실행에서 dynamic variant는 non-dynamic의
12회보다 48회 prompt rollout을 사용했고, 10개 group을 거부했으며, 1 update가
`exhausted`였다. 생성 token도 약 246에서 987~988로 증가했다. 성능만 비교하고 이
비용을 숨기면 공정하지 않다.

toy verifier reward는 정답 `1.0`, 형식은 맞지만 오답 `0.1`, invalid `0.0`이다.
논문의 대규모 binary correctness reward와 같다고 주장하지 않는다. Dynamic filter는
값 자체가 아니라 group 내 reward 변화 여부를 검사하므로 이 차이를 명시적으로
카드에 남긴다.

## 4. Token-level loss

GRPO의 response별 token 평균 후 sequence 평균 대신, batch의 모든 유효 action
token 합을 전체 유효 token 수로 나눈다.

\[
L_{token}=-\frac{1}{\sum_i |o_i|}\sum_i\sum_t
\min(\rho_{i,t}A_i,\operatorname{clip}(\rho_{i,t})A_i)
\]

prompt와 padding은 제외하고 response와 생성된 EOS만 action mask에 포함된다.
서로 다른 길이에서 sequence mean과 token mean이 다른 손계산 test를 제공한다.

## 5. Soft Overlong Reward Shaping

최대 response 길이를 `L`, buffer를 `B`, 실제 길이를 `l`이라 하면 추가 reward는

\[
r_{long}=-\alpha\,\operatorname{clip}
\left(\frac{l-(L-B)}{B},0,1\right)
\]

이다. `L-B`까지 0이고 buffer 안에서 선형 감소해 `L`에서 `-α`가 된다. 최대치를
넘는 입력도 `-α`로 clamp한다. 짧은 답을 무조건 선호시키는 목적이 아니라 length
boundary의 갑작스러운 truncation reward noise를 완화하는 장치다.

## 6. 무엇을 결론 내릴 수 없는가

C7 toy report의 overlong-only exact match `0.0625`, dynamic variant `0.03125`는
32개 validation prompt와 seed 하나의 관측값이다. 논문 AIME 결과를 재현하지 않고,
네 구성요소의 일반적 우열도 증명하지 않는다. 검증된 것은 다음뿐이다.

- 네 toggle이 서로 독립적인 수식·경로를 바꾼다.
- 동일 initial SFT hash와 non-dynamic prompt 순서를 검사한다.
- dynamic 추가 sampling, 거부, 예산 소진을 숨기지 않는다.
- checkpoint resume가 continuous 실행과 tensor 단위로 정확히 같다.

## Sources

- `dapo-2025`: 네 요소의 식과 동기
- `framework-verl`: 독립 라이선스 modern variant naming 교차검산
- `repo-dapo`: 존재와 논문 context만 기록; 구현 source로 사용하지 않음

---

[← GRPO 계열](grpo-family.md) · [강좌 지도](../course-map.md) · [L13 notebook](../../notebooks/ko/L13_dapo_variants.ipynb) · [다음: Agentic RL →](agentic-rl.md)
