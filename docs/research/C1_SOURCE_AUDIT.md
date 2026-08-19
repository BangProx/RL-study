# C1 문헌·코드·라이선스 감사

> 감사 기준일: 2026-08-19 (Asia/Seoul)
> 상태: C1 구현 입력 고정
> 기계 판독 원장: [`docs/sources.yml`](../sources.yml)

이 문서는 “유명한 구현이니까 가져온다”는 결정을 금지하고, RL-study가 실제로
참고할 근거와 라이선스 경계를 고정한다. 핵심 구현은 모두 논문과 공개 수식을
바탕으로 새로 작성하는 **clean-room PyTorch 구현**이다. upstream 저장소는
동작과 용어를 교차 검증하는 자료이며, 현재 복사하거나 변형한 upstream 파일은
없다.

## 1. 결론 먼저

- 저장소 라이선스는 승인된 **Apache-2.0**으로 고정한다.
- 교육용 `toy` 구현은 framework에 의존하지 않는 순수 PyTorch 코드로 만든다.
- `laptop` 실제 모델 adapter는 **TRL v1.10.0**, `server` adapter는
  **verl v0.9.0**을 사용한다. OpenRLHF v0.11.0은 비교 대안으로만 문서화한다.
- 실제 모델 preset은 SmolLM2-135M-Instruct, Qwen3-0.6B, Qwen3-4B 순으로
  laptop smoke, laptop quality, server 역할을 맡는다.
- 실제 검증 데이터는 GSM8K `main/train`을 쓰고, 공식 test는 평가에만 쓴다.
  UltraFeedback Binarized는 100MB가 훨씬 넘으므로 server 선택 항목이다.
- 승인된 외부 Agentic benchmark는 ALFWorld 0.4.2의 **text-only optional
  adapter**로 한정한다. 코드·데이터·solver를 vendor하지 않는다.
- DAPO 공식 레포에는 감사한 commit에서 LICENSE가 없다. 따라서 그 레포의
  코드, 설정, 문서 문장, 그림을 일절 복사·변형하지 않는다. DAPO 네 요소는
  논문을 기준으로 독립 구현하고 Apache-2.0인 verl의 공개 동작과 교차 검증한다.

## 2. 조사 방법과 주장 경계

근거는 최신 논문/책 → 저자·기관 공식 저장소의 고정 revision → 사용하는
framework의 공식 문서와 source → 기관 기술 글 순서로 판정했다. GitHub의
default branch, archive 상태, release/tag와 LICENSE 원문을 확인했고, Hugging
Face Hub API와 Dataset Viewer에서 model/dataset revision, file size, row 수,
split과 card license를 확인했다.

이 감사가 보장하지 않는 것도 명확하다.

- 논문에 보고된 대형 모델 성능은 이 저장소의 실행 결과가 아니다.
- Hub card의 license tag는 제3자 데이터의 모든 권리를 대신 보증하지 않는다.
- 고정 revision 이후의 upstream 변경은 2026-08-19 기준 “최신” 주장에
  포함되지 않는다.
- model weight, 대규모 dataset, benchmark asset은 저장소에 재배포하지 않는다.

## 3. 알고리즘별 근거와 구현 결정

| 주제 | 1차 근거 | 공식/기관 코드 감사 | 이 저장소의 결정 |
|---|---|---|---|
| PPO | PPO arXiv 1707.06347v2 | OpenAI Spinning Up, MIT, `038665d…` | clipped surrogate, value, entropy, GAE를 PyTorch로 독립 구현 |
| RLHF-PPO | Stiennon et al. v3, InstructGPT v1 | `summarize-from-feedback`는 archived·Modified MIT | SFT→RM→rollout→KL+PPO를 toy 규모로 독립 구현; upstream 데이터 미사용 |
| DPO | DPO arXiv 2305.18290v3 | 저자 repo Apache-2.0, `f8b8c0f…` | chosen/rejected reference log-ratio loss를 손계산 test와 함께 독립 구현 |
| GRPO | DeepSeekMath 2402.03300v3 | DeepSeek-Math code MIT, model은 별도 license | group-relative advantage, clip, per-token KL의 variant를 명시해 독립 구현 |
| RLOO | 2402.14740v2 | Papers record에 저자 공식 repo 없음; TRL/verl은 Apache-2.0 | leave-one-out baseline의 critic-free 목적함수를 논문 기반 구현 |
| DAPO | 2503.14476v2 | 공식 DAPO repo `33fe317…`에 LICENSE 없음 | **코드 참조 금지**. Clip-Higher, Dynamic Sampling, token loss, overlong shaping을 논문 기반 clean-room 구현 |
| Dr. GRPO | 2503.20783v2 | `sail-sg/understand-r1-zero`, MIT, `dfca49d…` | 길이 편향을 만드는 reduction과 보정 reduction을 나란히 구현 |
| GSPO | 2507.18071v2 | Papers record에 저자 공식 repo 없음; verl은 Apache-2.0 | token ratio와 구분되는 sequence ratio·clip·loss를 논문 기반 구현 |
| Agentic RL | Agent Lightning v1, Agent-R1 v2 | Agent Lightning v1.0.0 MIT; AgentR1 조직 repo MIT | step-level MDP, trace/transition 분리, token/tool mask, outcome/process credit를 작은 환경에서 구현 |

### 원 논문과 framework가 다를 때

이름만 같고 reduction이 다른 구현을 하나로 뭉개지 않는다. 각 loss 함수는
`paper_variant`, `trl_variant`, `verl_variant` 같은 출처 의미를 config와
algorithm card에 표시한다. 강좌의 기본값은 수식과 가장 직접 연결되는
`paper_variant`이고, framework adapter는 해당 framework의 config를 명시한다.
특히 GRPO 계열은 다음 축을 독립적으로 기록한다.

1. advantage가 reward standard deviation으로 정규화되는가,
2. token 평균·sequence 평균·고정 길이 분모 중 무엇을 쓰는가,
3. importance ratio와 clipping이 token-level인가 sequence-level인가,
4. KL을 loss에 넣는가 reward에 넣는가,
5. zero-variance group과 지나치게 긴 응답을 어떻게 처리하는가.

이 구분은 Dr. GRPO의 길이 편향과 GSPO의 sequence-level update를 “GRPO 옵션”
하나로 잘못 축약하지 않기 위해 필요하다.

## 4. Framework 선택

| 후보 | 고정 버전 | 확인된 범위 | 채택 역할 | 이유/제약 |
|---|---|---|---|---|
| TRL | v1.10.0 / `a7be897…` | SFT, Reward, DPO, experimental PPO, GRPO, RLOO, PEFT | laptop adapter | Transformers/PEFT와 자연스럽고 100M~1B one-step 경로가 단순함 |
| verl | v0.9.0 / `483b8a0…` | PPO, GRPO, RLOO, DAPO, DrGRPO, GSPO, multi-turn tool calling | server adapter | 승인된 최신 변형과 Agentic server recipe를 가장 적은 adapter 수로 연결 |
| OpenRLHF | v0.11.0 / `3c3be62…` | 분산 RLHF stack | 비교 대안 | 세 번째 adapter까지 유지하면 학습 효과보다 유지보수·CI 비용이 커서 실행 adapter에서는 제외 |
| Agent Lightning | v1.0.0 / `8f8b8f9…` | trace, trainer/execution separation | schema/architecture reference | core trainer dependency로 묶지 않고 optional interoperability 경계만 설계 |

`toy`의 수학 정답 구현은 위 framework 어느 것도 import하지 않는다. 따라서
네트워크 없는 CPU notebook과 unit test가 framework 버전 변화 때문에 깨지지
않는다.

## 5. 모델 preset과 다운로드 안전성

| preset | model/revision | weights | license | 현실적 용도 |
|---|---|---:|---|---|
| `laptop-smoke` | `HuggingFaceTB/SmolLM2-135M-Instruct@12fd25f…` | 269,060,552 B | Apache-2.0 | CPU/MPS/CUDA 실제 forward+backward 최소 검증 |
| `laptop-quality` | `Qwen/Qwen3-0.6B@c1899de…` | 1,503,300,328 B | Apache-2.0 | MPS/CUDA LoRA 짧은 학습; CPU는 smoke만 |
| `server` | `Qwen/Qwen3-4B@1cfa9a7…` | 8,044,982,000 B | Apache-2.0 | CUDA server recipe; 현 laptop에서는 external-manual |

세 모델 모두 weight가 100MB를 넘는다. loader는 model ID, revision, license,
예상 bytes, cache 경로를 먼저 출력한 뒤 `--accept-download`가 없으면 다운로드
전에 종료해야 한다. `trust_remote_code=False`가 기본이다. 현재 laptop의 실제
지원 여부는 marketing 표가 아니라 C8의 one-step capability probe 결과로
기록한다.

## 6. Dataset 결정과 split 정책

### GSM8K — laptop 실제 RLVR 기본

- ID/revision: `openai/gsm8k@740312add88f781978c0658806c59bc2815b9866`
- license: MIT; public/ungated
- `main/train`: 7,473 rows, parquet 2,306,545 bytes
- `main/test`: 1,319 rows, parquet 419,088 bytes
- schema: `question: string`, `answer: string`

학습은 `main/train`만 사용한다. seed 42로 prompt ID를 안정 정렬한 뒤 train에서
validation을 결정적으로 분리한다. 공식 test는 최종 평가 전용이며 reward
튜닝, early stopping, prompt 선택에 사용하지 않는다. laptop preference pair가
필요하면 train prompt에서 repository verifier로 응답을 생성하고 생성 config,
model revision과 verifier version을 lineage에 남긴다.

### UltraFeedback Binarized — server 선택 항목

- ID/revision:
  `HuggingFaceH4/ultrafeedback_binarized@3949bf5f8c17c394422ccfab0c31ea9c20bdeb85`
- license tag: MIT; public/ungated
- `train_prefs`: 61,135 rows, 225,891,836 bytes
- `test_prefs`: 2,000 rows, 7,291,160 bytes
- 전체: 187,405 rows, 649,967,196 bytes

`train_prefs`만으로도 100MB를 넘으므로 laptop 기본 다운로드에서 제외한다.
생성 preference 데이터이므로 card에 기록된 생성 model, filtering과 원 source
lineage를 실제 과학적 주장 전에 다시 검토한다. 단순한 MIT tag를 사람 데이터와
생성 output의 모든 권리 보증으로 확대 해석하지 않는다.

### TinyReasoning — 완전 오프라인 core

문제, 정답, 틀린 응답, preference, verifier를 동일 seed/config에서 생성한다.
생성 규칙과 split hash가 checksum 역할을 하며 원본 대형 데이터는 없다. 모든
알고리즘의 기본 실행과 notebook CI는 이 lineage를 공유한다.

## 7. 외부 Agentic benchmark

승인된 adapter 대상은 `alfworld/alfworld` 0.4.2이다. ALFWorld 자체와
TextWorld는 MIT지만 선택적 planner인 Fast Downward는 GPL-3.0이다. 따라서:

- core package와 toy Agentic 환경은 ALFWorld를 import하지 않는다.
- adapter는 설치 여부를 검사하고 설명적인 설치 안내만 제공한다.
- ALFWorld/TextWorld/Fast Downward 코드, asset, binary를 vendor하거나
  재배포하지 않는다.
- 기본 adapter는 text-only evaluation이다. full environment 설치는
  `external-manual`로 표시한다.
- macOS arm64에서는 upstream이 안내하는 x86 conda 제약까지 문서화하고,
  이 laptop에서 실행하지 않은 상태를 성공처럼 기록하지 않는다.

이 선택은 CalculatorToolEnv와 LocalLookupEnv라는 완전 오프라인 학습 환경 두
개를 대체하지 않는다. 외부 benchmark는 일반화 확인을 위한 선택 경로다.

## 8. 라이선스 판정과 NOTICE 정책

| 대상 | 판정 | 허용한 사용 |
|---|---|---|
| RL-study 새 코드·문서 | Apache-2.0 | 저장소 전체 기본 license |
| MIT/Apache upstream | 호환 | 수식·동작 교차 검증; 현재 copied/adapted file 없음 |
| summarize-from-feedback | Modified MIT, archived | 역사/구조 참고만; 데이터·코드 미배포 |
| DAPO 공식 repo | license 없음 | URL·commit·존재와 논문 context만 기록; 코드/문서/assets 참조 구현 금지 |
| DeepSeek model weights | code와 별도 조건 | 실제 사용하는 Hub model card의 license만 적용 |
| ALFWorld stack | MIT + optional GPL-3.0 | optional external 설치만; vendor 없음 |

향후 upstream 코드를 복사하거나 실질적으로 변형하는 변경이 생기면 PR 전에
`docs/sources.yml`의 provenance를 `copied` 또는 `adapted`로 바꾸고, 원 license
text·copyright·NOTICE 요구를 `NOTICE`와 `third_party/` 원장에 추가해야 한다.
지금의 NOTICE는 “third-party source files are not redistributed”라는 현재
상태를 정직하게 기록한다.

## 9. A1~A8 승인 반영

| 승인 | 고정 결정 | 단계 |
|---|---|---|
| A1 | Apache-2.0 | 1단계 |
| A2 | 한국어 L00~L16 완료·검증 뒤 영어 mirror | 2단계 포함 승인 |
| A3 | Colab quickstart | 1단계 |
| A4 | MkDocs site | 2단계 포함 승인 |
| A5 | RLOO, Dr. GRPO, GSPO 모두 구현 | 1단계 |
| A6 | 대화형 비교 UI | 2단계 포함 승인 |
| A7 | Linux/macOS/Windows CPU CI + scheduled network/link/notebook CI | 전체 승인 |
| A8 | ALFWorld external Agentic benchmark adapter | 승인 |

추가 범위는 이번 감사에서 승인하지도 구현하지도 않았다.

## 10. C2로 넘기는 고정 입력

1. core 수학 구현은 clean-room PyTorch이고 모든 tensor shape·gradient·reduction
   선택을 test와 algorithm card에 연결한다.
2. `toy`는 offline, `laptop`은 TRL, `server`는 verl adapter 경계로 설계한다.
3. model/dataset은 exact revision을 생략할 수 없고 100MB guard를 우회하지
   않는다.
4. DAPO는 paper-only clean-room 경계를 code review checklist에 넣는다.
5. benchmark/서버/CUDA 미실행 결과는 `external-manual`로 표현한다.
6. 논문 수치, upstream 수치, local 실행 수치를 artifact schema에서 서로 다른
   필드로 저장한다.
