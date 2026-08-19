# RL-study 구현 Goal 계약서

> 상태: **Goal 실행 중 / C11 로컬 완료, hosted Colab·CI 승인 대기**
> 작성일: 2026-08-19 (Asia/Seoul)
> 로컬 경로: `/Users/bangbyeonghun/Documents/nlp/RL-study`
> 대상 원격: `https://github.com/BangProx/RL-study.git`

이 문서는 Codex의 장기 실행 `/goal`에 전달할 유일한 구현 계약서다. 단순한
할 일 목록이 아니라 하나의 목표, 변경 금지 범위, 승인 지점, 체크포인트,
검증 방법과 종료 조건을 정의한다.

작성 직전 로컬 디렉터리는 비어 있었고 현재는 이 계약서만 추가된 상태이며,
아직 Git 저장소로 초기화되지 않았다. 원격 저장소의 존재·이력·기본 브랜치·권한은
구현 시작 시 C0에서 읽기 전용으로 먼저 확인한다.

작성일의 1차 `git ls-remote` 점검에서는 대상 URL이 정상 응답했고 ref가 하나도
없었다. 즉 원격은 현재 빈 저장소로 보이지만, C0에서 초기화 직전에 다시
확인하고 상태가 달라졌다면 새 이력을 덮어쓰지 않는다.

---

## 0. Goal 시작 방법

### 시작 전 조건

1. 이 문서의 **승인 게이트 A** 결정이 모두 기록되어 있는지 확인한다.
2. 구현 시작 직전에 원격 상태가 달라지지 않았는지 다시 확인한다.
3. 아래 명령을 그대로 사용한다.

```text
/goal GOAL.md를 유일한 구현 계약으로 삼아 RL-study를 완성하라. 승인된 범위만 구현하고 C0부터 C12까지 각 체크포인트의 검증을 통과하라. 모든 완료 조건에 대한 실행 증거가 생길 때까지 계속 진행하되, 승인되지 않은 범위 확장, 라이선스 충돌, 원격 이력 충돌, 유료 서비스 사용, 또는 검증할 수 없는 연구 주장을 만나면 임의로 결정하지 말고 멈춰 보고하라.
```

Goal 상태를 확인할 때는 `/goal`, 필요하면 `/goal pause`, `/goal resume`,
`/goal clear`를 사용한다. 이 계약은 공식 OpenAI Goal 지침의 “하나의 목표와
종료 조건, 선행 문서, 검증 루프, 체크포인트별 진행 기록” 원칙을 따른다.

- 공식 Goal 안내: <https://learn.chatgpt.com/use-cases/follow-goals>

---

## 1. 단일 목표와 범위 해석

### 단일 목표

초심자가 이 저장소 하나만으로 **LLM에 쓰이는 강화학습을 이해하고, 핵심
알고리즘을 수식에서 PyTorch 코드로 직접 연결하고, 노트북과 CLI에서 실제로
학습·평가하며, 노트북용 toy 실험에서 서버용 실제 모델 실험까지 확장**할 수
있는 한국어 중심의 self-contained 오픈소스 강좌와 참조 구현을 완성한다.

### “RL을 전부 안다”의 정직한 범위

이 저장소가 보장하는 범위는 다음과 같다.

- LLM RL 논문과 코드를 읽는 데 필요한 확률·최적화·RL 기초를 외부 강의 없이
  익힌다.
- bandit, MDP, Bellman equation, dynamic programming, Monte Carlo, TD,
  Q-learning, DQN, policy gradient, REINFORCE, actor-critic, GAE, PPO를 코드로
  연결한다.
- LLM을 policy로 보는 관점, reward/preference model, RLHF, DPO, GRPO,
  DAPO와 Agentic RL을 end-to-end로 구현한다.
- offline RL, model-based RL, continuous control, multi-agent RL, imitation
  learning 등 인접 분야는 전체 지형과 LLM RL과의 관계를 설명하되, 모든 분야의
  대규모 benchmark를 재현한다고 주장하지 않는다.
- “최신”은 구현 시점에 다시 조사한 문헌 기준일로 한정하며, 미래의 모든
  논문까지 포함한다는 뜻으로 쓰지 않는다.

### 대상 독자

- Python 문법과 기본적인 tensor 연산은 알지만 RL은 처음인 학습자
- 수식만 읽으면 흐름을 놓치고 직접 실행한 코드와 출력으로 이해하는 학습자
- 긴 집중보다 짧은 학습 단위, 즉시 피드백, 명확한 현재 위치가 필요한 학습자
- laptop에서 시작해 이후 GPU 서버 실험으로 확장하려는 연구자·개발자

---

## 2. 완료의 정의

아래 항목이 **모두** 참일 때만 Goal을 완료한다.

1. 한국어 강좌 notebook 17개와, 한국어판 검증 후 만드는 영어 mirror notebook
   17개가 같은 학습 목표·수식·코드·실험·검사를 제공한다.
2. 모든 한국어·영어 기본 notebook은 이전 notebook의 kernel 상태 없이 깨끗한
   환경에서 위에서 아래로 실행된다.
3. 빠른 경로는 약 4~6시간, 전체 경로는 약 11~14시간 안에 마칠 수 있고,
   실제 실행 시간과 학습자 예상 시간을 README에 기록한다.
4. 외부 다운로드가 없는 `toy` 프로필에서 bandit부터 DAPO와 Agentic RL까지
   핵심 루프를 CPU로 실행할 수 있다.
5. `src/rl_study`에 tabular Q-learning, DQN, REINFORCE, actor-critic, GAE,
   PPO, reward model, RLHF-PPO, DPO, GRPO, DAPO의 교육용 clean-room 구현이
   있으며 CLI에서 train/eval/checkpoint/resume이 가능하다.
6. classic PPO와 LLM RLHF-PPO의 state/action/reward/ratio/mask 차이를 코드,
   수식, 테스트로 구분한다.
7. PPO, DPO, GRPO, DAPO는 논문의 식과 코드 line/function을 연결한
   implementation note, 대안, trade-off, 흔한 오류와 최소 한 개의 ablation을
   가진다.
8. preference data 생성 또는 로드 → reward model 또는 verifier → rollout →
   advantage/reward → policy update → evaluation의 전체 RLHF/RLVR 파이프라인을
   실제로 실행한다.
9. Agentic RL은 최소 두 개의 완전 오프라인 multi-turn tool environment와 C1
   감사에서 선정한 외부 benchmark adapter에서 trajectory 수집, action/tool
   mask, 종료 조건, outcome/process reward, credit assignment와 policy
   update를 실제로 수행한다.
10. `toy`, `laptop`, `server` 프로필이 분리되고, model/dataset/algorithm/device를
    config와 CLI로 선택할 수 있다.
11. laptop 프로필에는 실제 공개 causal LM을 LoRA로 최소 한 step 이상 학습하는
    검증된 preset이 있고, 메모리·다운로드·예상 시간·지원 device가 명시된다.
    무료 Colab quickstart는 새 runtime에서 toy 경로를 실행하고 accelerator가
    있으면 실제 모델 one-step smoke까지 수행한다.
12. server 프로필에는 실제 분산 RL framework로 확장 가능한 pinned adapter와
    recipe가 있으나, 실행하지 않은 GPU 결과를 통과로 표시하지 않는다.
13. `python -m rl_study.demo --profile toy`가 10분 이내를 목표로 작은 비교
    실험을 실행하고 JSON, PNG, HTML report와 checkpoint를 만든다. 한국어 core
    완료 뒤에는 같은 artifact를 탐색하는 대화형 비교 UI도 제공한다.
14. 모든 결과는 seed, config, git commit, environment, wall time, peak memory,
    model/dataset revision을 담은 experiment card로 추적된다.
15. 수식·주장·구현 선택마다 논문, 저자 공식 코드, framework 공식 문서 중
    하나 이상의 추적 가능한 출처가 있다.
16. 가져오거나 변형한 코드는 라이선스가 호환되고 NOTICE와 provenance가
    보존된다. 라이선스가 불명확하면 복사하지 않고 논문 기반 clean-room으로
    재구현한다.
17. unit/integration/notebook tests, lint, type check, 한영 parity, local link
    check, Linux/macOS/Windows CPU CI와 주기적 network/link/notebook job이 모두
    통과한다.
18. README와 MkDocs 강의 사이트에서 설치, 15분 quickstart, 강좌 지도,
    하드웨어별 실행, 모델 교체, 문제 해결, 연구 결과의 한계와 기여 방법을 바로
    찾을 수 있다.
19. `origin`이 대상 GitHub 저장소를 가리키고, 기존 원격 이력을 덮어쓰거나
    force-push하지 않은 상태로 공개 가능한 repository hygiene를 갖춘다.
20. 요구사항과 산출물·테스트를 연결한 traceability matrix에서 모든 필수
    항목이 `verified` 또는 근거가 있는 `external-manual` 상태다.

GitHub star 100k는 직접 통제하거나 검증할 수 있는 완료 조건이 아니다. 대신
정확성, 학습 경험, 재현성, 접근성, 검색 가능성, 문서 품질, 기여 용이성을
그 수준을 지향하는 품질 기준으로 사용한다.

---

## 3. 승인 게이트 A — 결정 완료

아래 항목은 원 요구보다 범위를 넓히거나 유지비·실행비에 영향을 주므로 먼저
사용자 결정을 받았다. 2026-08-19에 모든 항목이 승인되었으며, 단계 표기가 있는
항목은 한국어 core 검증 뒤 2단계로 구현한다. 이 표의 결정은 Goal 범위에
구속력을 가진다.

| ID | 제안 | 권장안 | 이유와 비용 | 결정 |
|---|---|---|---|---|
| A1 | 저장소 라이선스 | Apache-2.0 | 특허 조항과 연구·상업 재사용 명확성이 좋다. third-party는 별도 감사가 필요하다. | **승인 — Apache-2.0** |
| A2 | 영어 mirror 강좌 | 한국어 완성 후 2단계 추가 | 국제 접근성과 기여 가능성은 크게 높지만 notebook·문서 동기화 비용이 거의 2배다. | **승인 — 한국어 완성 후 2단계** |
| A3 | 무료 Colab quickstart | 추가 | 로컬 설치 실패를 줄이고 laptop 사양이 낮아도 첫 실험이 가능하다. Colab 환경 drift를 주기적으로 점검해야 한다. | **승인** |
| A4 | MkDocs 정적 강의 사이트 | 2단계 추가 | 검색·탐색·SEO·공유가 쉬워진다. notebook과 사이트 내용의 중복 방지 자동화가 필요하다. | **승인 — 2단계** |
| A5 | 최신 비교 알고리즘 | RLOO, Dr. GRPO, GSPO를 L12~L13의 심화 구현으로 추가 | GRPO/DAPO가 무엇을 고친 것인지 더 정확히 비교할 수 있다. 강좌 약 90분과 테스트 범위가 늘어난다. 최신 버전·공식 코드·라이선스를 C1에서 재검증한다. | **전부 승인** |
| A6 | 대화형 실험 비교 UI | 필수 정적 HTML report 완성 후 2단계 추가 | 학습 전후 응답, reward, KL, entropy, clip fraction을 filter하며 비교해 ADHD 학습자의 피드백 루프가 짧아진다. 별도 UI dependency와 유지비가 생긴다. | **승인 — 2단계** |
| A7 | 확장 CI | 3-OS CPU 필수 + 주기적 network/link/notebook job | 이식성과 문서 부패를 줄인다. hosted runner 시간과 외부 서비스 drift가 늘어난다. | **승인** |
| A8 | 외부 Agentic benchmark adapter | 오프라인 toy 2개는 필수, ALFWorld/WebShop류는 2단계 | toy를 넘어 실제 agent 연구로 확장할 수 있지만 dataset, simulator, 라이선스, 설치 비용이 크다. | **승인** |

### 결정 기록 형식

2026-08-19 사용자 승인 기록:

```text
A1 승인
A2 한국어 완성 후 2단계 승인
A3 승인
A4 2단계 승인
A5 전부 승인
A6 2단계 승인
A7 승인
A8 승인
```

구현 중 새로운 추가 아이디어가 생기면 다음 다섯 항목을 먼저 보고한다.

1. 무엇을 추가하려는가
2. 학습 효과 또는 유지보수상 이유
3. 예상 구현·검증 비용
4. 기존 범위에서 빠지거나 늦어질 것
5. 승인/보류/대체 선택지

사용자 승인이 오기 전에는 해당 변경을 구현하지 않는다. 다만 기존 필수 범위의
버그 수정, 테스트 보강, 문서 오탈자 수정은 추가 기능으로 보지 않는다.

---

## 4. 고정 요구사항

승인 게이트와 무관하게 아래는 필수다.

### 콘텐츠

- 확률, 기대값, 분산, sampling, log-probability, entropy, cross-entropy,
  KL divergence, gradient와 automatic differentiation
- autoregressive LM, tokenization, causal mask, sequence log-probability,
  decoding temperature와 stop condition
- RL의 agent/environment/state/observation/action/reward/return/policy/value,
  episode, horizon, discount, on-policy/off-policy
- exploration/exploitation, credit assignment, bootstrapping, bias/variance
- bandit, MDP, Bellman equation, policy/value iteration, Monte Carlo, TD,
  Q-learning, function approximation, replay와 target network
- policy gradient theorem의 직관과 REINFORCE, baseline, actor-critic, GAE, PPO
- SFT, preference data, reward modeling, RLHF, KL regularization, RLAIF와 RLVR의
  차이
- PPO, DPO, GRPO, DAPO의 논문 충실 핵심 구현
- reward hacking, specification gaming, length bias, entropy collapse,
  KL explosion, overoptimization, data leakage와 evaluation contamination
- single-turn reasoning RL과 multi-turn Agentic RL의 trajectory·credit 차이
- 교육용 축약과 논문 규모 재현의 명확한 구분

### 구현 설명

각 핵심 알고리즘은 다음 질문에 답해야 한다.

- 이 수식의 각 기호가 코드의 어떤 tensor이며 shape는 무엇인가?
- gradient는 어디로 흐르고 어디에서 끊기는가?
- 왜 이 reduction, normalization, mask, clipping을 사용했는가?
- 원 논문과 공식 구현이 다른 경우 무엇이 다르고 어느 쪽을 선택했는가?
- 선택 가능한 대안은 무엇이며 안정성·메모리·속도 trade-off는 무엇인가?
- 틀리기 쉬운 구현은 무엇이고 어떤 test가 그 오류를 잡는가?
- toy 결과가 논문의 대형 모델 결과와 어떻게 다르게 해석되어야 하는가?

### 실행 환경

- Python 3.10~3.12와 최신 안정 PyTorch를 기준으로 하되 exact version은 C1에서
  호환성 확인 후 lock한다.
- 기본 설치는 CPU와 네트워크 없는 toy 실행에 무거운 RL framework를 요구하지
  않는다.
- macOS CPU/MPS, Linux CPU/CUDA, Windows CPU/CUDA의 지원 여부를 기능별로
  정직하게 표시한다.
- CUDA-only library는 optional extra와 server profile에 격리한다.
- 요청한 device가 없을 때 조용히 CPU로 바꾸지 않는다. 사용자가 명시적으로
  fallback을 허용한 경우에만 전환한다.
- 경로는 `pathlib`, 임시 파일은 `tempfile`, process entry는 Windows `spawn`
  규칙을 지킨다.
- API key, cloud GPU, 유료 judge, 외부 tracking 계정은 core 경로에 필요 없다.

### 다운로드 안전장치

- model/dataset을 받기 전에 ID, revision, license, 예상 다운로드 크기, cache
  위치와 예상 메모리를 보여준다.
- 100MB가 넘는 asset은 명시적인 `--accept-download` 없이는 받지 않는다.
- `trust_remote_code=false`가 기본이며 꼭 필요하면 별도 위험 설명과 승인을
  요구한다.
- dataset test split은 학습에 쓰지 않는다.
- checkpoint, cache, model weights, API key와 원본 대규모 dataset은 Git에
  commit하지 않는다.

---

## 5. 학습 경험 계약

### 강좌 길이와 경로

- 1단계 17개 한국어 notebook, 한국어 검증 후 2단계 17개 영어 mirror notebook
- lesson당 약 25~60분, 5~8분 micro-section으로 분할
- **빠른 경로:** 표시된 필수 셀만 약 4~6시간
- **전체 경로:** 모든 설명·실습·심화 체크 약 11~14시간
- **연구 경로:** server recipe와 최신 논문 ablation은 별도이며 전체 시간에
  포함하지 않는다.

실제 첫 실행 후 예상 시간이 25% 이상 어긋나면 notebook 수를 무작정 늘리지
말고 section을 재분배하고 측정값을 갱신한다.

### 모든 notebook의 고정 리듬

1. **이번 lesson의 목표:** 최대 3개
2. **현재 위치:** 전체 개념 지도에서 이전/현재/다음 한 줄
3. **먼저 예측:** 실행 결과를 10~30초 생각하는 질문
4. **짧은 설명:** 한 문단 또는 작은 그림
5. **한 셀 한 개념:** 눈에 보이는 숫자·표·plot·trajectory 출력
6. **즉시 확인:** assertion 또는 1~3분 회상 문제
7. **왜 이렇게 구현했나:** 대안과 trade-off
8. **흔한 함정:** 틀린 코드/수식 → 원인 → 수정 → 관련 test
9. **쉬어가기 지점:** 지금 닫아도 되는 안전한 checkpoint
10. **60초 요약:** 핵심 3줄
11. **다음 연결:** 다음 lesson에서 해결할 질문 하나

### ADHD와 접근성 원칙

- 긴 문단, 거대한 코드 셀, 끝없는 progress bar와 경고 출력을 피한다.
- 코드 셀은 보통 25줄 이하이고 하나의 질문에만 답한다.
- section 시작에 예상 시간과 남은 section 수를 표시한다.
- 필수/심화/서버 전용 경로를 일관된 label로 구분한다.
- 색만으로 의미를 구분하지 않고 선 모양, marker, 직접 label을 함께 쓴다.
- 모든 그림에 alt text와 한 줄 텍스트 결론을 제공한다.
- 답은 학습자가 먼저 시도할 공간 뒤의 접힌 `<details>`에 둔다.
- notebook마다 2~4개의 “내가 자주 틀리는 것” 오답노트를 둔다.
- noisy animation, 자동 재생 media, 과도한 장식과 불필요한 선택지를 넣지 않는다.
- 실패한 실험도 숨기지 않고 “무엇을 관찰하면 되는가”를 짧게 안내한다.

### Notebook 구조

각 notebook은 아래 level-2 section 순서를 지킨다.

1. `## Goal`
2. `## Setup`
3. `## Steps`
4. `## Checks`
5. `## 내가 자주 틀리는 것`
6. `## 60초 요약`
7. `## Next Steps`
8. `## Sources`

`Setup`은 seed, device, profile, package version, network 필요 여부를 출력한다.
`Checks`는 단순 정답 문장이 아니라 최소 하나의 실행 가능한 assertion을 가진다.

### 한영 mirror 계약

- 한국어판을 먼저 기술 검토하고 top-to-bottom 검증한 뒤 영어판을 만든다.
- 두 언어판의 lesson/cell stable ID, code cell, 수식, seed, config, figure data,
  assertion과 source ID는 동일해야 한다.
- 번역 markdown은 자연스러운 학습 흐름을 우선하되 개념의 난이도와 주장을
  바꾸지 않는다.
- CI에서 lesson 순서, code hash, exercise/check 수, figure와 source parity를
  검사한다.
- 영어판 추가 때문에 한국어 core의 버그 수정이나 최신 문헌 반영이 중단되지
  않도록 한 소스에서 생성 가능한 공통 metadata와 code cell을 사용한다.

---

## 6. 강좌 지도

| # | 제목 | 핵심 산출물 | 전체 시간 | 경로 |
|---:|---|---|---:|---|
| 00 | 15분에 보는 LLM RL 전체 지도 | 작은 policy가 reward로 변하는 첫 curve와 전체 개념 지도 | 25분 | 빠른/전체 |
| 01 | 확률·미분·PyTorch 생존 키트 | categorical distribution, log-prob, entropy, KL, gradient 직접 계산 | 40분 | 빠른/전체 |
| 02 | 언어모델은 어떻게 확률을 내는가 | tiny causal LM, token/sequence log-prob, mask와 sampling | 45분 | 전체 |
| 03 | 가장 작은 RL: bandit | exploration/exploitation, reward, regret, policy update | 35분 | 빠른/전체 |
| 04 | MDP와 Bellman equation | GridWorld에서 policy/value iteration | 50분 | 빠른/전체 |
| 05 | 경험으로 배우기: MC·TD·Q-learning | 같은 환경에서 backup target과 bias/variance 비교 | 50분 | 전체 |
| 06 | DQN과 function approximation | replay, target network, detach, instability ablation | 50분 | 전체 |
| 07 | Policy Gradient와 REINFORCE | log-derivative trick, return, baseline, variance 시각화 | 50분 | 빠른/전체 |
| 08 | Actor-Critic·GAE·PPO from scratch | advantage, ratio, clip, value/entropy loss와 mini PPO trainer | 60분 | 빠른/전체 |
| 09 | LLM을 policy로 보기 | token action, sequence reward, preference, verifier, KL reference | 50분 | 빠른/전체 |
| 10 | RLHF-PPO end-to-end | SFT → preference → reward model → rollout → PPO update | 60분 | 빠른/전체 |
| 11 | DPO: RL 없이 보이는 RL 목적함수 | chosen/rejected log-ratio, reference model, beta와 label noise | 50분 | 빠른/전체 |
| 12 | GRPO와 verifiable reward | group-relative advantage, critic 제거, reward variance와 bias | 55분 | 빠른/전체 |
| 13 | DAPO와 안정적인 reasoning RL | asymmetric clip, dynamic sampling, token loss, overlong shaping | 55분 | 빠른/전체 |
| 14 | 실제 소형 모델 학습과 서버 확장 | LoRA train, checkpoint/resume, model 선택, server recipe | 55분 | 전체/연구 |
| 15 | Agentic RL: tool을 쓰며 여러 turn 학습하기 | step-level MDP, tool env, masks, process/outcome reward, credit assignment | 60분 | 빠른/전체 |
| 16 | 실패 진단·평가·capstone | reward hacking, KL/entropy/clip 진단, 공정 비교와 experiment card | 55분 | 빠른/전체 |

### L00 개념 지도 요구사항

Mermaid와 동등한 ASCII fallback을 함께 제공한다.

```text
확률·최적화
  → bandit → MDP/Bellman → MC/TD/Q-learning → DQN
  → policy gradient → actor-critic/GAE → PPO
  → LLM policy + preference/reward
       ├─ RLHF-PPO
       ├─ DPO (offline preference objective)
       ├─ GRPO/RLVR → DAPO
       └─ Agentic RL: observation → action/tool → feedback → next step
모든 경로 → evaluation, reward hacking 진단, reproducibility
```

각 lesson은 이 지도에서 현재 node와 바로 앞·뒤 node만 다시 보여준다.

---

## 7. 교육용 데이터와 환경

### 완전 오프라인 toy suite

고정 seed로 코드에서 생성하고, 생성 규칙·split·정답을 테스트한다.

| 이름 | 역할 | 필수 관찰 |
|---|---|---|
| `BernoulliBandit` | exploration과 policy update | regret와 action probability |
| `TinyGridWorld` | MDP, Bellman, MC/TD/Q/DQN | value error와 success rate |
| `TinyReasoning` | SFT/RM/RLHF/DPO/GRPO/DAPO 공통 LLM task | exact match, format, length, reward |
| `CalculatorToolEnv` | multi-turn tool Agentic RL | tool validity, step reward, task success |
| `LocalLookupEnv` | offline retrieval/tool Agentic RL | citation correctness, unnecessary calls, success |

`TinyReasoning`은 작은 정수 산술·규칙 추론 문제, 정답 verifier, 맞고 틀린
응답, preference pair를 하나의 lineage에서 생성한다. 알고리즘 비교에는 동일한
prompt IDs, 초기 weight, token budget, eval split을 쓴다.

toy LLM은 고정 vocabulary의 tokenizer와 작은 decoder-only Transformer를
저장소 코드로 구현한다. 실제 parameter 수, 한 step 시간과 peak memory를
runtime에서 출력한다. 구체적인 layer/dimension/dataset row 수는 C2에서 laptop
측정 후 고정하고 변경 이유를 기록한다.

### 실제 dataset

구현 전에 C1에서 license, revision, card, row 수, compressed size, 사용 split,
test contamination 위험을 감사한다. 기본 후보는 작은 verifiable reasoning
dataset과 재배포 가능한 preference dataset이다. 감사 전에는 특정 dataset을
필수로 확정하거나 자동 다운로드하지 않는다.

외부 dataset은 다음 조건을 만족해야 한다.

- 공식 card 또는 원 출처에서 라이선스를 확인할 수 있다.
- train/validation/test 정책이 명확하다.
- laptop smoke subset을 deterministic하게 고를 수 있다.
- 원 논문 결과와 이 저장소의 축약 결과를 혼동하지 않는다.
- 생성 preference/reward라면 생성 model, prompt, verifier와 filter lineage를
  기록한다.

---

## 8. 실행 프로필과 모델 선택

### `toy` — 기본·오프라인·CPU

- 저장소 내 생성 data, tabular model과 tiny Transformer
- network 및 계정 불필요
- 모든 notebook과 core CI의 기본 경로
- 모든 필수 알고리즘의 forward/loss/update smoke 실행
- 짧은 실제 train, checkpoint, resume, eval과 report 생성

### `laptop` — 실제 공개 모델

- 구현 시점에 검증한 100M~1B급 causal LM을 기본 preset으로 선정한다.
- 최소 초경량 preset과 품질 우선 preset을 구분한다.
- Transformers + PEFT LoRA를 기본으로 하고 QLoRA는 지원 platform에서만
  별도 optional preset으로 둔다.
- CPU/MPS/CUDA capability를 실제 forward+backward one-step으로 probe한다.
- 짧은 context, small batch, gradient accumulation/checkpointing과 teacher/ref
  model offload 옵션을 제공한다.
- 임의 Hugging Face model ID도 받을 수 있지만 causal LM, tokenizer/chat
  template, value head, padding/EOS 호환성을 preflight한다.

### `server` — 논문형 확장

- multi-GPU framework는 core에서 분리한 thin adapter/recipe로 연결한다.
- 구현 시점의 공식 TRL, verl, OpenRLHF 등에서 필수 알고리즘 지원, 유지 상태,
  라이선스와 exact commit을 비교한 뒤 최소 조합을 선택한다.
- model, rollout engine, trainer, reward/verifier와 dataset을 config로 교체한다.
- laptop과 같은 metric/config 의미를 최대한 유지하되 동일하지 않은 차이를
  명시한다.
- 서버 hardware가 없으면 schema/unit test까지만 `verified`, 실제 train은
  `external-manual`로 남기며 성공 출력이나 성능을 만들지 않는다.

### 설정 계약

모든 실험은 검증된 typed config를 사용하고 unknown key를 오류로 처리한다.

```yaml
schema_version: 1
profile: toy                 # toy | laptop | server
algorithm:
  name: grpo                 # q_learning | dqn | reinforce | ppo | ...
  gamma: 0.99
  clip_low: 0.2
  clip_high: 0.2
  kl_coefficient: 0.0
data:
  id: tiny_reasoning
  revision: generated-v1
  seed: 42
model:
  policy: tiny-lm
  reference: tiny-lm-frozen
  reward: deterministic-verifier
training:
  seed: 42
  steps: 100
  batch_size: 16
  response_token_budget: 32768
  device: auto
  allow_device_fallback: false
evaluation:
  every_steps: 25
output:
  root: artifacts
```

환경변수가 config 값을 조용히 덮어쓰지 않으며, resume 시 비교 공정성에 영향을
주는 immutable field가 달라지면 실패한다.

---

## 9. 기술 아키텍처

### 제안 디렉터리

승인 결과에 따라 선택 디렉터리는 빠질 수 있으나 책임 분리는 유지한다.

```text
RL-study/
├── README.md
├── GOAL.md
├── PROGRESS.md
├── pyproject.toml
├── LICENSE
├── NOTICE
├── CITATION.cff
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── mkdocs.yml
├── notebooks/
│   ├── ko/00_...ipynb ... 16_...ipynb
│   ├── en/00_...ipynb ... 16_...ipynb
│   └── colab/quickstart.ipynb
├── src/rl_study/
│   ├── math/
│   ├── data/
│   ├── envs/
│   ├── models/
│   ├── algorithms/
│   │   ├── tabular.py
│   │   ├── dqn.py
│   │   ├── reinforce.py
│   │   ├── actor_critic.py
│   │   ├── ppo.py
│   │   ├── reward_model.py
│   │   ├── rlhf_ppo.py
│   │   ├── dpo.py
│   │   ├── grpo.py
│   │   └── dapo.py
│   ├── agentic/
│   ├── training/
│   ├── evaluation/
│   ├── diagnostics/
│   ├── ui/
│   └── cli.py
├── configs/{toy,laptop,server}/
├── docs/
│   ├── design/
│   ├── research/
│   ├── algorithms/
│   ├── math.md
│   ├── glossary.md
│   ├── hardware.md
│   ├── troubleshooting.md
│   └── sources.yml
├── tests/{unit,integration,notebooks}/
├── scripts/
├── artifacts/experiment-cards/
└── .github/
```

### 두 구현 층

1. **교육용 reference layer**
   - 순수 NumPy/PyTorch 중심의 짧고 읽기 쉬운 clean-room 구현
   - 논문 식과 직접 대응하고 모든 기본 notebook과 toy test가 사용
   - framework trainer 내부에 핵심 수학을 숨기지 않음
2. **research adapter layer**
   - Transformers/PEFT/TRL/verl 등 공식 framework와 연결
   - version/commit을 pin하고 reference layer와 loss parity를 작은 tensor로 검사
   - CUDA·분산 의존성은 optional import로 격리

### Notebook과 package의 중복 방지

- package 코드가 canonical 구현이다.
- notebook은 핵심 loss 또는 update를 10~25줄로 다시 만들고 작은 tensor에서
  package 결과와 같음을 assert한다.
- 긴 trainer를 notebook에 복사하지 않고, source link와 해당 function을
  보여주는 짧은 code view를 제공한다.
- notebook 간 hidden state나 수동으로 복사한 checkpoint에 의존하지 않는다.

### 공통 data contract

실제 타입은 구현 중 구체화하되 아래 의미를 보존한다.

```python
@dataclass(frozen=True)
class Transition:
    observation: Tensor | StructuredObservation
    action: Tensor | StructuredAction
    reward: Tensor
    next_observation: Tensor | StructuredObservation
    terminated: Tensor
    truncated: Tensor

@dataclass(frozen=True)
class TokenTrajectoryBatch:
    prompt_ids: Tensor             # int64 [B, P]
    response_ids: Tensor           # int64 [B, T]
    attention_mask: Tensor         # bool [B, P+T]
    action_mask: Tensor            # bool [B, T]
    old_logprobs: Tensor | None    # float [B, T], detached
    reference_logprobs: Tensor | None
    values: Tensor | None
    rewards: Tensor                # [B] or [B, T]
    advantages: Tensor | None
    returns: Tensor | None
    episode_ids: Tensor
    step_ids: Tensor | None
    terminated: Tensor
```

Agentic step은 observation, model action의 원 token IDs, parsed tool call,
environment output, reward, termination과 다음 observation을 함께 보존한다.
`Token → text → Token` 재토큰화가 원 token 경계를 바꾸지 않도록 테스트한다.

### CLI 계약

최소한 다음 흐름을 제공한다.

```text
python -m rl_study.demo --profile toy
rl-study train --config configs/toy/ppo.yaml
rl-study train --config configs/toy/grpo.yaml
rl-study eval --checkpoint <path>
rl-study inspect-run <artifact-directory>
rl-study preflight --profile laptop --model <model-id>
```

모든 명령은 `--help`, dry-run 또는 preflight, 명확한 오류 메시지와 non-zero
exit code를 제공한다.

---

## 10. 알고리즘 구현 계약

### 공통 공정성

비교 실험은 다음을 공유한다.

- serialized initial policy hash
- prompt/example IDs와 split hash
- optimizer와 scheduler, 최대 optimizer steps
- 처리한 environment steps 또는 response token budget
- decoding parameters와 평가 함수
- seed set, checkpoint cadence와 stopping rule

학습 방식별 계산량이 본질적으로 다르면 동일하다고 꾸미지 않고 model forward,
generated tokens, wall time와 peak memory를 함께 보고한다. 특정 알고리즘이 항상
이겨야 한다는 완료 조건을 두지 않는다.

### Tabular RL / DQN

- terminal과 truncation을 구분한다.
- Bellman target에서 terminal bootstrap을 제거한다.
- Q-learning의 off-policy target과 behavior policy를 분리한다.
- DQN target은 detach/no-grad이고 target network update 방식(hard/soft)을
  선택 가능하게 한다.
- replay sampling, epsilon schedule, target sync와 seed를 기록한다.
- Double DQN, prioritized replay 등은 대안으로 설명하되 승인된 범위만 구현한다.

### REINFORCE / Actor-Critic / GAE

- `log_prob * return`의 부호, reward-to-go와 episode return 차이를 테스트한다.
- baseline은 expectation을 바꾸지 않으면서 variance를 줄이는 이유를 작은 실험으로
  보인다.
- actor loss에서 advantage를 필요에 따라 detach한다.
- GAE의 `gamma`, `lambda`, terminal/truncation mask와 마지막 bootstrap을
  analytic trajectory로 검산한다.

### PPO

- rollout policy(`old`)와 update policy(`current`)를 분리한다.
- importance ratio, clipped surrogate, approximate KL, clip fraction, entropy,
  value loss를 각각 log한다.
- advantage normalization, value clipping, entropy bonus, KL early stopping,
  minibatch/epoch 수는 옵션이며 선택 이유를 설명한다.
- ratio=1, positive/negative advantage, clip boundary를 analytic test로 검산한다.
- classic continuous/discrete PPO와 token-level LLM PPO가 공유하는 수학과 다른
  reduction을 분리한다.

### Reward model / RLHF-PPO

- preference pair의 chosen/rejected ordering과 padding/mask를 검증한다.
- reward model loss, calibration, held-out preference accuracy와 length shortcut을
  진단한다.
- policy reward는 task/reward-model score, reference KL penalty, optional
  token-level shaping을 구분해 기록한다.
- prompt token, response token, EOS, padding, tool/environment token이 어떤
  loss와 reward에 포함되는지 truth table을 제공한다.
- policy, reference, reward, value model의 gradient 소유권과 eval/train mode를
  테스트한다.
- adaptive/fixed KL controller와 whitening/reward normalization의 대안을 비교한다.

### DPO

- policy와 frozen reference의 chosen/rejected sequence log-ratio를 명시한다.
- response token만 합산하고 prompt/pad를 제외한다.
- beta의 의미, reference-free 옵션의 의미와 위험, label noise의 영향을 보인다.
- DPO가 online rollout RL은 아니지만 KL-constrained reward objective에서
  유도되는 관계를 정확히 설명한다.
- 원 논문 reference 구현과 현대 framework 구현의 reduction 차이를 감사한다.

### GRPO

- prompt당 group sampling과 group-relative advantage를 구현한다.
- reward 표준편차 0인 group, small group, duplicated completion을 안전하게 다룬다.
- critic 제거가 메모리를 줄이는 대신 group sampling과 reward variance에
  의존하는 trade-off를 보인다.
- old/current/reference log-prob, importance ratio, clipping, KL 항의 정의를
  구현 variant별로 구분한다.
- token/sequence normalization과 length bias를 ablation한다.
- DeepSeekMath/후속 구현에서 같은 “GRPO” 이름 아래 달라진 부분을 숨기지 않는다.

### DAPO

논문에서 제시한 네 요소를 독립 toggle과 ablation으로 구현한다.

1. Clip-Higher / asymmetric clipping
2. Dynamic Sampling
3. Token-level Policy Gradient Loss
4. Overlong Reward Shaping

dynamic sampling이 모두 맞거나 모두 틀린 group을 처리하는 방식, sampling 예산,
길이 normalization, EOS/overlong mask, asymmetric clip의 positive/negative
advantage 동작을 테스트한다. toy 구현은 대규모 AIME 성능 재현이 아니라 네
기법의 동작을 관찰하기 위한 것임을 명시한다.

### Agentic RL

- single response를 한 action으로 보는 방식과 step-level MDP를 모두 설명한다.
- environment의 `reset/step`, observation visibility, tool schema, action parser,
  timeout, invalid action, termination/truncation을 typed interface로 둔다.
- outcome reward와 process/step reward를 분리하고 double counting을 막는다.
- response/action token만 policy loss에 포함하고 environment/tool output은
  기본적으로 mask한다.
- final reward를 모든 token/step에 복사하는 baseline과 return/GAE 또는 승인된
  credit assignment를 비교한다.
- context truncation, stale trajectory, asynchronous rollout, tool nondeterminism,
  retokenization drift와 reward hacking을 failure lesson에서 재현한다.
- 외부 command/tool은 sandbox와 allowlist를 사용하고 toy environment는 임의
  shell/network를 실행하지 않는다.

---

## 11. 연구 출처와 라이선스

### 근거 우선순위

1. peer-reviewed 최종 논문 또는 최신 arXiv 원문
2. 저자·연구기관의 공식 repository와 해당 commit/tag
3. 사용하는 framework의 공식 문서와 source code
4. 저자·연구기관 기술 블로그
5. 보조 설명 자료

블로그, 영상, 개인 구현은 발견과 직관 보조에는 쓸 수 있으나 핵심 수식이나
성능 수치의 유일한 근거로 사용하지 않는다.

### 시작점 문헌

아래는 C1 조사의 시작점이며, 구현 시 최신 version·정정·공식 코드·license를
다시 확인한다.

- Sutton & Barto, *Reinforcement Learning: An Introduction (2nd ed.)*
  - <http://incompleteideas.net/book/the-book-2nd.html>
- Mnih et al., *Playing Atari with Deep Reinforcement Learning*
  - <https://arxiv.org/abs/1312.5602>
- Schulman et al., *High-Dimensional Continuous Control Using GAE*
  - <https://arxiv.org/abs/1506.02438>
- Schulman et al., *Proximal Policy Optimization Algorithms*
  - <https://arxiv.org/abs/1707.06347>
  - <https://github.com/openai/spinningup>
- Stiennon et al., *Learning to Summarize with Human Feedback*
  - <https://arxiv.org/abs/2009.01325>
  - <https://github.com/openai/summarize-from-feedback>
- Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback*
  - <https://arxiv.org/abs/2203.02155>
- Rafailov et al., *Direct Preference Optimization*
  - <https://arxiv.org/abs/2305.18290>
  - <https://github.com/eric-mitchell/direct-preference-optimization>
- Shao et al., *DeepSeekMath* (GRPO)
  - <https://arxiv.org/abs/2402.03300>
  - <https://github.com/deepseek-ai/DeepSeek-Math>
- Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*
  - <https://arxiv.org/abs/2503.14476>
  - <https://github.com/BytedTsinghua-SIA/DAPO>
- Luo et al., *Agent Lightning*
  - <https://arxiv.org/abs/2508.03680>
  - <https://github.com/microsoft/agent-lightning>
- Cheng et al., *Agent-R1*
  - <https://arxiv.org/abs/2511.14460>
  - <https://github.com/AgentR1/Agent-R1>

### Source manifest

각 알고리즘·dataset·model마다 `docs/sources.yml`과 사람이 읽는 audit 문서에
다음을 기록한다.

- title, authors, venue/status, arXiv/version, 확인 날짜
- 공식 repository URL, exact commit SHA 또는 release/tag
- SPDX license와 확인한 LICENSE 경로
- 실제로 참고한 equation/algorithm/file/function
- `copied`, `adapted`, `clean-room-reimplemented` 분류
- upstream과 우리 구현의 의도적 차이
- 논문 보고 결과, upstream 결과, 이 저장소 실행 결과의 구분
- model/dataset ID, exact revision, size, split, checksum과 redistribution 조건

라이선스가 없거나 충돌하면 코드를 복사·vendor하지 않는다. 논문 설명만으로
clean-room 구현 가능한지 검토하고, 불가능하면 구현을 멈춘 뒤 사용자에게
근거와 대안을 보고한다.

---

## 12. 검증 전략

### 수학·단위 테스트

- 알려진 작은 categorical distribution의 entropy, CE, FKL/RKL, log-ratio
- Bellman update와 terminal/truncation
- MC/TD/Q-learning target
- replay/target network detach
- REINFORCE sign과 reward-to-go
- GAE analytic trajectory와 lambda 경계
- PPO ratio/clip/KL/value/entropy cases
- chosen/rejected ordering과 sequence mask
- reward model pairwise loss와 calibration input
- RLHF reward+KL decomposition
- DPO loss를 손계산한 batch와 비교
- GRPO group normalization과 zero-variance group
- DAPO asymmetric clip, dynamic filter, token reduction, overlong shaping
- prompt/response/pad/EOS/tool/environment mask truth table
- teacher/reference/reward model parameter와 gradient 불변성

### 통합 테스트

- 모든 필수 알고리즘의 one-step finite loss와 의도한 parameter update
- toy train → checkpoint → resume → eval
- 같은 seed에서 중단 후 resume와 연속 run의 허용 오차 내 parity
- 공정 비교 config의 initial hash, data IDs와 budget audit
- Agentic env reset/step/invalid action/termination/timeout
- toy policy가 두 offline agent environment에서 trajectory와 update를 생성
- `rl_study.demo`가 JSON/PNG/HTML/checkpoint를 생성
- model/device/dataset mismatch가 학습 전 설명적인 오류로 종료
- 명시적 동의 없이 100MB 초과 download나 remote code 실행을 시도하지 않음

stochastic 학습은 특정 알고리즘의 승리를 강제하지 않는다. analytic test,
NaN/Inf 방지, 여러 seed의 넓은 sanity bound와 회귀 추세를 결합한다.

### Notebook 검증

- `nbformat` 구조와 stable lesson/cell metadata
- 고정 section 순서, 목표 수, predict/check/mistake/source cell 존재
- toy profile top-to-bottom clean execution
- traceback, 거대 output, debug dump와 progress spam 부재
- markdown 결론과 실제 출력 일치
- figure alt text, 직접 label과 텍스트 요약
- network/server cell은 명시적 tag와 deterministic toy fallback 보유
- 실행하지 않은 optional 셀 결과를 결론 근거로 쓰지 않음

### 권장 검증 명령 형태

구현 도구가 확정되면 README와 CI에 실제 명령으로 고정한다.

```bash
python -m pytest
python -m ruff check .
python -m mypy src
python scripts/execute_notebooks.py --profile toy --language ko
python scripts/execute_notebooks.py --profile toy --language en
python scripts/check_notebook_contract.py
python scripts/check_bilingual_parity.py
python scripts/check_links.py --local
python -m rl_study.demo --profile toy --non-interactive
python -m mkdocs build --strict
```

### Experiment card

각 실제 실행은 다음을 남긴다.

- git commit, dirty 여부, config hash, dependency lock hash
- OS, Python, PyTorch, CPU/GPU/MPS, RAM/VRAM, dtype
- model/dataset exact revision과 license manifest ID
- seed, split hash, decoding과 metric 정의
- optimizer/environment steps, processed/generated token 수
- wall-clock, peak memory, checkpoint와 artifact 경로
- 결과, seed 분산, 알려진 편차와 실패
- paper-reported 수치인지 local-executed 수치인지 표시

---

## 13. 구현 체크포인트

Goal은 아래 순서로 진행하고 매 단계마다 `PROGRESS.md`를 갱신한다. 상태 보고는
현재 checkpoint, 만든 산출물, 실행한 검증, 남은 일, blocker만 짧게 쓴다.

### C0. 안전한 저장소 부트스트랩

- 로컬 경로·파일·Git 상태와 부모 지침을 다시 확인한다.
- 대상 GitHub 원격의 존재, visibility, default branch와 기존 이력을 읽기
  전용으로 확인한다.
- 원격이 비어 있으면 local Git을 `main`으로 초기화하고 `origin`을 설정한다.
- 원격 이력이 있으면 먼저 fetch/clone 전략과 충돌을 보고하고 안전한 통합 없이
  덮어쓰지 않는다.
- force-push, history 삭제와 사용자 파일 덮어쓰기를 금지한다.

검증: 실제 `pwd`, `git status`, branch, `git remote -v`, 원격 이력 요약.

### C1. 최신 문헌·코드·라이선스 감사와 승인 고정

- 필수 논문 최신 version, 공식 repo, commit, license를 확인한다.
- PPO/DPO/GRPO/DAPO와 Agentic RL 공식 구현의 실제 차이를 표로 만든다.
- A1~A8 승인 결정을 문서와 `PROGRESS.md`에 기록한다.
- RLOO, Dr. GRPO, GSPO의 논문·공식 코드·license를 감사한다.
- model/dataset 후보의 size, license, device별 가능성을 제시하고 preset을 고정한다.
- 새로운 추가 제안은 사용자 승인 전 구현하지 않는다.

검증: `docs/research/C1_SOURCE_AUDIT.md`, `docs/sources.yml`, 승인 기록.

### C2. 교육·기술 설계 고정

- 17개 lesson별 objective, prerequisite, demo, exercise, source를 고정한다.
- notebook style/metadata, glossary, concept map과 빠른 경로를 고정한다.
- typed data/config/API, 공정 비교, model compatibility와 artifact schema를 정한다.
- 20개 완료 조건의 traceability matrix를 만든다.
- toy model/data 크기를 실제 laptop timing 후 고정한다.

검증: 모든 요구가 lesson, code module, test 중 하나 이상에 연결됨.

### C3. Package 기반과 수학

- packaging, typed config, seed/device, logging, atomic checkpoint 기반
- probability/KL/masked reduction과 tiny tokenizer/causal LM
- bandit, GridWorld, TinyReasoning generator
- unit tests와 최소 CLI skeleton

검증: core import가 network/CUDA optional dependency 없이 되고 수학 test 통과.

### C4. 고전 RL reference 구현

- policy/value iteration, MC, TD, Q-learning, DQN
- REINFORCE, actor-critic, GAE, classic PPO
- 동일 환경 비교와 instability ablation
- algorithm별 equation-to-code 문서

검증: analytic/unit tests, 각 알고리즘 train/resume/eval toy smoke.

### C5. LLM policy·preference·reward 기반

- sequence/action mask와 token log-prob
- SFT baseline, preference data, deterministic verifier와 reward model
- frozen reference/value model 계약과 metric
- length/format shortcut 진단

검증: pairwise loss, mask, gradient ownership과 작은 reward model train.

### C6. RLHF-PPO와 DPO

- toy end-to-end RLHF-PPO와 DPO trainer
- old/current/reference/reward/value 분리
- 공정한 SFT/RM/RLHF/DPO 비교
- 실패·대안 ablation과 checkpoint/resume

검증: 손계산 loss parity, integration train, experiment cards.

### C7. GRPO·DAPO와 승인된 최신 변형

- group rollout, advantage, clipping/KL/reduction choices
- DAPO 네 요소와 독립 ablation
- RLOO, Dr. GRPO, GSPO의 승인된 심화 구현과 문헌 비교
- 논문·framework 간 variant naming을 명시

검증: analytic edge cases, toy learning run과 구성요소별 report.

### C8. Laptop·server profile

- 실제 model/dataset loader, download guard, preflight
- LoRA와 지원될 때만 QLoRA
- model override, checkpoint/resume와 resource estimate
- pinned research framework adapter/recipe

검증: 가능한 local device에서 실제 공개 모델 one-step 이상. 검증하지 못한
hardware는 exact manual command와 `external-manual` 상태를 기록.

### C9. Agentic RL

- calculator와 local lookup environment
- step-level trajectory, parser, mask, reward와 credit assignment
- baseline과 policy update 비교
- C1에서 선정한 외부 benchmark adapter

검증: deterministic environment tests, end-to-end multi-turn train/eval, failure cases.

### C10. 한국어·영어 notebook과 Colab

- 한국어 L00~L16을 순서대로 작성하되 package test를 함께 추가한다.
- 각 notebook은 고정 리듬, 빠른/전체 경로, 오답노트와 source를 지킨다.
- 한국어판은 작성 즉시 toy top-to-bottom 실행하고 결론을 출력에 맞춰 쓴다.
- 한국어 17개가 검증된 뒤 영어 mirror 17개와 parity checker를 완성한다.
- Colab quickstart는 새 runtime에서 clone → install → toy demo → 선택형 실제
  model smoke 순서로 실행한다.

검증: 한영 notebook contract/parity checker, 전체 clean execution manifest와
날짜가 기록된 Colab 실행 증거.

### C11. 문서·접근성·CI·공개 품질

- README quickstart/course map/hardware/troubleshooting
- algorithm cards, math/glossary, provenance, community health files
- Linux/macOS/Windows CPU CI와 주기적 network/link/notebook job
- 필수 정적 report, MkDocs strict build와 대화형 비교 UI

검증: 신규 사용자의 documented journey를 clean environment에서 재실행.

### C12. 최종 재현 감사와 release 준비

- clean clone/install → demo → tests → 한영 notebook 전체 실행
- Colab, MkDocs site와 대화형 UI의 사용자 경로 재검증
- 20개 완료 조건과 traceability evidence 대조
- 실행하지 않은 claim/output, stale link, secret, 큰 file, license 누락 감사
- 원격 동기화 전 diff, branch와 commit scope 확인
- 버전, CHANGELOG, CITATION과 known limitations 작성

검증: 최종 명령·환경·시간·결과 manifest와 `PROGRESS.md` 완료 기록. 원격 push나
release 생성은 사용자가 별도로 승인한 범위에서만 수행한다.

---

## 14. 진행·중단·재개 규칙

- 각 checkpoint가 끝나면 테스트 결과와 artifact 경로를 `PROGRESS.md`에
  추가하고 다음 단계로 간다.
- 일시적 테스트 실패는 원인과 다음 시도를 기록하고 계속 해결한다.
- 동일한 외부 blocker가 반복되고 안전한 대안이 없을 때만 blocked로 보고한다.
- 사용자 선택, 유료 자원, credential, 라이선스 판단, 원격 이력 충돌이 필요하면
  해당 경계를 넘지 않고 멈춘다.
- 목표가 긴 이유만으로 범위를 줄이거나 미검증 항목을 완료 처리하지 않는다.
- context가 압축되어도 `GOAL.md`, `PROGRESS.md`, 현재 git diff와 test 결과를
  읽고 이어서 진행하며 이미 완료한 일을 처음부터 반복하지 않는다.
- 새 논문이 발견되어도 필수 알고리즘을 조용히 교체하지 않는다. 출처, 효과,
  비용과 기존 범위 영향을 먼저 보고하고 승인을 기다린다.

---

## 15. 금지 사항과 비목표

- star 수, 논문 benchmark와 특정 성능 수치를 보장하지 않는다.
- 대형 GPU 논문 결과를 laptop toy 결과로 “재현”했다고 표현하지 않는다.
- 실행하지 않은 cell output, speed, memory, score와 graph를 만들지 않는다.
- 라이선스 불명 코드를 복사하거나 upstream repo 전체를 무분별하게 vendor하지
  않는다.
- API key, token, credential, private data를 출력·commit하지 않는다.
- 사용자 승인 없이 유료 API, cloud GPU, model/dataset upload, public release,
  원격 push를 수행하지 않는다.
- test set 학습, seed cherry-picking, 유리한 metric만 보고하는 방식을 금지한다.
- DPO를 online RL과 동일하다고 하거나 GRPO/DAPO variant 차이를 숨기지 않는다.
- unsupported device/QLoRA를 지원한다고 표시하거나 OOM 위험이 있는 full
  fine-tuning으로 조용히 fallback하지 않는다.
- Agentic tool environment가 임의 shell/network 명령을 실행하게 두지 않는다.
- notebook이 이전 실행 순서, 전역 kernel 상태 또는 사용자의 수동 파일 편집에
  의존하지 않는다.

---

## 16. 완료 보고 형식

Goal 완료 보고는 홍보 문구보다 증거를 우선한다.

1. 한 문장 결과
2. 구현된 강좌·알고리즘·프로필 요약
3. 빠른 시작 명령과 첫 notebook 링크
4. 실제 실행한 검증과 결과 manifest
5. laptop/server 검증 상태와 실행하지 못한 항목
6. 알려진 제한, 연구 문헌 기준일과 보류된 승인 항목
7. Git branch/remote/commit 상태와 사용자가 다음에 할 수 있는 선택

모든 필수 작업이 실제로 끝나고 검증 증거가 있을 때만 Goal을 완료로 표시한다.
