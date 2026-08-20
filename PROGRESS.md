# RL-study Goal 진행 기록

## 2026-08-19 — Goal 시작 및 C0 완료

- 현재 checkpoint: 완료
- 로컬 경로: `/Users/bangbyeonghun/Documents/nlp/RL-study`
- 승인: A1~A8 전체 승인
  - 1단계: Apache-2.0, Colab, RLOO/Dr. GRPO/GSPO, 확장 CI, 외부 Agentic benchmark
  - 한국어 core 후 2단계: 영어 mirror, MkDocs, 대화형 비교 UI
- 원격 재확인: `git ls-remote --symref`가 exit 0, ref 없음
- Git: 새 저장소, branch `main`, 아직 commit 없음
- origin: `https://github.com/BangProx/RL-study.git` (fetch/push)
- 안전 조건: 기존 원격 이력이 없음을 확인했고 force-push나 원격 쓰기를 하지 않음
- 환경 메모: GitHub CLI `gh`는 설치되어 있지 않아 권한 확인은 추후 push 전 수행
- 검증: 원격 ref 없음, local `main`, `origin` URL과 clean bootstrap 조건 확인

## 2026-08-19 — C1 최신 문헌·코드·라이선스 감사 완료

- 현재 checkpoint: C2 교육·기술 설계
- 산출물:
  - `docs/research/C1_SOURCE_AUDIT.md`
  - `docs/sources.yml` (JSON-compatible YAML 1.2)
  - `LICENSE`, `NOTICE`, `.gitignore`, 초기 `README.md`
- 고정 결정:
  - clean-room PyTorch 교육 구현 + TRL v1.10.0 laptop adapter + verl v0.9.0
    server adapter
  - SmolLM2-135M-Instruct / Qwen3-0.6B / Qwen3-4B preset
  - GSM8K train 기반 실제 RLVR, UltraFeedback Binarized는 server 선택 항목
  - ALFWorld 0.4.2 text-only optional adapter
- 라이선스 경계: DAPO 공식 repo에는 감사한 commit에서 license가 없어 코드,
  config, 문서, asset을 복사·변형하지 않음. 논문 clean-room 구현만 허용
- 검증:
  - `python3 -m json.tool docs/sources.yml` 통과
  - manifest assertion: papers=14, repositories=12, models=3, datasets=3
  - 모든 pinned repository의 `copied_files=[]` 확인
  - `git diff --check` 통과
- 다음 작업: 17개 lesson, notebook metadata/style, typed API/config, artifact schema,
  공정 비교 계약과 완료 조건 traceability를 설계하고 toy 크기를 laptop에서 측정

## 2026-08-19 — C2 교육·기술 설계 완료

- 현재 checkpoint: C3 Package 기반과 수학
- 산출물:
  - `docs/design/curriculum.md`: L00~L16 objective/prerequisite/demo/exercise/source
  - `docs/design/notebook-style.md`: ADHD 학습 리듬, metadata, 접근성, 한영 parity
  - `docs/design/architecture.md`: typed batch/protocol, strict config, device/download,
    fairness, checkpoint, experiment-card와 adapter schema
  - `docs/design/traceability.md`: 완료 조건 R01~R20과 알고리즘/test 연결
  - `docs/glossary.md`
  - `scripts/benchmark_toy_sizes.py`, `docs/research/C2_TOY_BENCHMARK.json`
- 환경 고정: project `.venv` Python 3.10.12, PyTorch 2.13.0
- 실제 CPU 측정(M4 host, 3 warm-up + 20 measured step):
  - `tiny-micro-v1` 20,224 params, median 3.081ms
  - `tiny-base-v1` 77,312 params, median 6.430ms
  - canonical `tiny-v1` 242,976 params, median 9.598ms, p95 10.809ms
- MPS 판정: build=True지만 현재 sandbox runtime에서 available=False. 성능 수치를
  만들지 않았고 C8 capability probe 대상으로 유지
- 검증:
  - `python3 -m json.tool`로 source/benchmark manifest 통과
  - `scripts/check_design_contract.py` 통과
  - lessons=17, requirements=20, source IDs=33, tiny-v1 params=242,976
- 다음 작업: packaging, strict config, seed/device, tensor/math, tiny tokenizer/LM,
  offline 환경/데이터, atomic checkpoint와 CLI skeleton을 unit test와 함께 구현

## 2026-08-19 — C3 Package 기반과 수학 완료

- 현재 checkpoint: C4 고전 RL reference 구현
- 환경: `.venv` Python 3.10.12, torch 2.13.0, NumPy 2.2.6, PyYAML 6.0.3
- 산출물:
  - `pyproject.toml`, editable `rl-study` CLI/package
  - immutable tensor/environment types, strict YAML config/hash, seed와 명시적 device
  - categorical entropy/cross-entropy/forward KL, selected log-prob, masked reductions
  - printable-ASCII tokenizer, canonical `tiny-v1` causal LM(242,976 params)
  - BernoulliBandit, TinyGridWorld, TinyReasoning 256/64/128 + preference 512
  - integrity hash와 atomic rename을 쓰는 checkpoint/RNG/resume 기반, JSONL metric
  - `python -m rl_study.demo` foundation smoke와 CLI preflight/train dry-run
- 첫 test 실패와 수정: canonical 128-token tokenizer를 64-vocab micro model에
  잘못 연결한 통합 test를 발견해 실제 경로인 `tiny-v1`으로 contract를 맞춤
- 검증:
  - ruff: pass
  - strict mypy: 23 source files, no issues
  - pytest: 31 passed in 3.15s
  - `pip check`: no broken requirements
  - core import 후 Transformers/TRL/verl/ALFWorld 미로딩 확인
  - design contract와 CLI GRPO config dry-run 통과
  - foundation demo split hash:
    `sha256:f6811f784261cbd0d8895519ebe6fd468c5882a0d5f156d28dac9b9abb7fe186`
- 다음 작업: DP/MC/TD/Q-learning, DQN, REINFORCE, actor-critic, GAE, classic PPO의
  수식 대응 loss/update, train/eval/checkpoint/resume와 instability ablation 구현

## 2026-08-19 — C4 고전 RL reference 구현 완료

- 현재 checkpoint: C5 LLM policy·preference·reward 기반
- 산출물:
  - `algorithms/tabular.py`: value/policy iteration, MC, TD(0), Q-learning
  - `algorithms/dqn.py`: replay, frozen target, hard/soft sync, Double DQN option
  - `algorithms/policy_gradient.py`: REINFORCE, running baseline, actor-critic
  - `math/returns.py`: discounted return, terminated/truncated-aware GAE
  - `algorithms/ppo.py`: old/current ratio, asymmetric bounds, KL/clip metrics
  - 5개 classic CLI train/eval/checkpoint/resume runner와 toy config
  - `docs/algorithms/classic.md`, `docs/research/C4_CLASSIC_BENCHMARK.json`
- local CPU 기본 결과(seed 42, TinyGridWorld):
  - Q-learning/DQN/REINFORCE/actor-critic/PPO validation success 모두 1.0
  - DQN frozen-initial-target ablation success 0.0 (이 toy seed의 failure demo)
  - value/policy iteration 7회, value max diff 0
  - MC visited-state RMSE 0, TD(0) 100 episodes RMSE 7.44e-6
- 해석 경계: 위 수치는 local toy sanity이며 논문 재현이나 일반 알고리즘 순위가 아님
- resume 증거:
  - Q-learning, DQN(replay+target 포함), REINFORCE(baseline 포함), actor-critic,
    PPO 모두 20 continuous == 10+checkpoint+10, state tensor `rtol=0, atol=0`
  - 실제 CLI Q-learning step 20 → resume step 40 → eval success 1.0
  - `training.steps` 연장은 허용, model/data/algorithm immutable hash 변경은 거부
- 검증: ruff pass, strict mypy 30 source files, pytest 56 passed in 3.43s,
  JSON manifest와 `pip check` 통과
- 다음 작업: response/action mask, token/sequence log-prob, SFT, deterministic
  verifier, preference batch와 reward model, frozen model/length shortcut 진단 구현

## 2026-08-19 — C5 LLM policy·preference·reward 기반 완료

- 현재 checkpoint: C6 RLHF-PPO와 DPO
- 산출물:
  - causal next-token 위치에 정렬된 prompt/action/attention mask와 sequence log-prob
  - response+EOS만 최적화하는 SFT loss/trainer
  - correct=1.0 / valid-format-wrong=0.1 / invalid=0 deterministic verifier
  - Bradley–Terry pairwise reward loss, tiny reward/value model
  - reference/reward freeze+eval+gradient-none 계약과 parameter SHA-256
  - held-out numeric/format accuracy와 score-length Pearson shortcut 진단
  - `docs/algorithms/llm-foundations.md`, C5 실행 JSON
- context 수정: 기존 prompt+response가 64 token을 넘을 수 있어 prompt를 의미가
  같은 짧은 형식으로 변경. 전체 combined 56~60 token, silent truncation 없음
- 현재 TinyReasoning split hash:
  `sha256:f238657bbf6c0a112debf7ef3ffafb452c14308dfb5ce57d9abe4f77ac1deedd`
- local CPU 실행:
  - SFT 100 step: 첫/마지막 10 loss 16.121→0.311,
    validation response-token accuracy 0.902, 1.88s
  - reward model 120 step: loss 0.446→0.268, held-out preference 0.797
  - numeric 0.594 vs format 1.0, score-length correlation +0.832
- 해석: RM은 format/length shortcut이 강함. 성공 지표로 숨기지 않고 C6 policy
  update에서 reward exploitation, exact verifier, length를 분리 추적
- 검증: ruff pass, strict mypy 37 source files, pytest 69 passed in 4.25s,
  design/source manifest 검증 통과
- 다음 작업: old/current/reference/reward/value 역할을 분리한 toy RLHF-PPO,
  DPO hand-parity loss, fair SFT/RM/RLHF/DPO 실행과 checkpoint/resume 구현

## 2026-08-19 — C6 RLHF-PPO와 DPO 완료

- 현재 checkpoint: C7 GRPO·DAPO와 승인된 최신 변형
- 산출물:
  - `algorithms/dpo.py`: chosen/rejected policy-reference sequence log-ratio,
    response-only 합, beta와 label smoothing, frozen reference
  - `algorithms/rlhf_ppo.py`: old/current/reference/reward/value 분리, sampled token
    KL, terminal score, masked token return와 clipped PPO
  - DPO/RLHF CLI train/eval, policy+value optimizer를 포함한 atomic resume checkpoint
  - split/prompt hash, optimizer step, generated/processed token, model forward와
    peak memory를 기록하는 versioned experiment card
  - `docs/algorithms/{dpo,rlhf-ppo}.md`, 공정 비교 실행 JSON과 runner
- 공정 비교(seed 42, 같은 SFT policy hash, 같은 64 prompt occurrences, 각 8 step):
  - SFT greedy exact 0.0 / format 1.0
  - DPO preference accuracy 0.805 / exact 0.0 / format 0.875,
    offline response token 2,520 / model forward 36
  - RM-RLHF exact 0.0 / format 0.0, generated token 1,368 / forward 2,128
  - verifier-RLHF exact 0.031 / format 1.0, generated token 1,317 / forward 2,005
- 실패 해석: held-out RM preference accuracy 0.797만 보면 좋아 보이지만 length
  correlation +0.832인 shortcut이 policy update에서 format 붕괴로 증폭됐다. 이
  결과를 숨기거나 알고리즘 순위로 해석하지 않고 failure case로 고정
- resume 증거:
  - DPO 4 continuous == 2+resume+2, RLHF 2 continuous == 1+resume+1,
    전체 wrapper state `rtol=0, atol=0`
  - 실제 CLI DPO step 2→4→eval, RLHF step 1→2→eval/inspect-run 성공
- 검증: ruff pass, strict mypy 44 source files, pytest 81 passed in 31.03s,
  design contract, source/C6 JSON, `pip check` 통과
- 다음 작업: GRPO group-relative advantage와 edge case부터 구현하고 DAPO 네 요소,
  RLOO/Dr. GRPO/GSPO를 서로 다른 variant 이름과 ablation으로 연결

## 2026-08-19 — C7 GRPO·DAPO와 승인된 최신 변형 완료

- 현재 checkpoint: C8 Laptop·server profile
- 산출물:
  - `algorithms/grpo.py`: group mean/population-std advantage, token ratio/clip,
    pointwise k3 KL, sequence/token/Dr. GRPO reduction, RLOO와 GSPO objective
  - `algorithms/dapo.py`: Clip-Higher, Dynamic Sampling, global token loss,
    soft overlong shaping의 paper-only clean-room 구현
  - 다섯 알고리즘의 toy online trainer, frozen old/reference, strict config,
    CLI train/eval/checkpoint/resume와 versioned experiment card
  - `docs/algorithms/{grpo-family,dapo}.md`, 10-way component 비교 실행 JSON
- variant 경계:
  - GRPO는 group reward mean/std + token ratio + sequence별 token 평균 + k3 KL
  - RLOO는 completion 전체를 한 action으로 보는 leave-one-out REINFORCE
  - Dr. GRPO는 reward std와 response-length normalization을 제거한 고정 분모
  - GSPO는 token log-ratio의 평균을 exp한 sequence ratio와 sequence clipping
  - DAPO 네 요소를 none/각 하나/all로 독립 toggle
- 수치 안정성: 동일 weight Transformer clone도 CPU forward에서 약 `1e-6` log-prob
  차이가 생길 수 있어 k3에 `expm1`과 `1e-5` clone-equality guard를 적용하고
  값·gradient가 정확히 0인지 test로 고정
- 공정 비교(seed 42, 3 rollout step, group 4):
  - 10개 variant가 같은 SFT policy hash에서 시작
  - non-dynamic variant는 같은 ordered prompt hash, 12 rollout / 246 token
  - dynamic DAPO는 48 rollout / 987~988 token, rejected group 10,
    bounded budget exhausted update 1을 숨기지 않고 기록
  - overlong-only exact 0.0625, dynamic exact 0.03125는 32 prompt·단일 seed의
    관측값일 뿐 우승 또는 논문 규모 재현으로 해석하지 않음
- resume/CLI 증거:
  - GRPO/DAPO/RLOO/Dr. GRPO/GSPO 모두 2 continuous == 1+resume+1,
    전체 wrapper tensor `rtol=0, atol=0`
  - shell CLI GRPO step 1→2→eval/inspect-run, DAPO all-four step 1 성공
- 검증: ruff pass, strict mypy 49 source files, pytest 107 passed in 129.12s,
  design contract와 source/C7 JSON, `git diff --check` 통과
- 환경 메모: PyTorch 2.13 external device backend 자동 탐색이 이 Mac에서 매우
  느려지는 현상을 분리 진단했고 CPU toy test에는
  `TORCH_DEVICE_BACKEND_AUTOLOAD=0`을 사용. C8 troubleshooting/preflight에 반영 예정
- 다음 작업: 공개 LM/dataset loader와 explicit download guard, local LoRA one-step,
  resource estimator, pinned TRL/verl laptop/server recipe 구현

## 2026-08-19 — C8 Laptop·server profile 완료

- 현재 checkpoint: C9 Agentic RL
- offline contract:
  - audited model/dataset manifest, exact revision cache probe와 100MB download guard
  - arbitrary model은 revision/license/expected bytes 없으면 Hub lookup 전 실패
  - CPU/MPS/CUDA tensor probe와 device별 dtype, QLoRA CUDA+bitsandbytes hard gate
  - base/adapter/gradient/Adam/activation/runtime을 분리한 resource estimator
  - GSM8K `main/train` deterministic validation split, 공식 test contamination 차단
- 실제 optional stack:
  - TRL 1.10.0, Transformers 4.57.6, PEFT 0.20.0, Datasets 4.8.5,
    Accelerate 1.14.0; `requirements/laptop.lock` SHA-256 `cf45c4b…321934`
  - SmolLM2-135M-Instruct exact revision와 GSM8K exact revision을 명시 승인 후 download
  - cache 사용량 관측: model 260MB, dataset 4.5MB; 어떤 weight/data도 repo에 commit 안 함
- local CPU 실제 LoRA SFT:
  - q/v projection rank 8, trainable 460,800 / total 134,975,808 (`0.3414%`)
  - 2 optimizer step, completion token 103, model forward 10
  - last train loss 1.0795, train-derived 8-example validation loss 1.5466
  - peak RSS 1,258,356,736 bytes; 초기 estimator가 낮았던 failure를 숨기지 않고
    768MiB runtime overhead 항을 추가해 보수적으로 보정
  - continuous 2와 1+resume+1 adapter가 byte-identical,
    SHA-256 `0c56d05d…c0890f9`; offline eval step 2 성공
- server 경계:
  - verl v0.9.0 / `483b8a0…`, Qwen3-4B / `1cfa9a7…` DAPO·GSPO
    Hydra recipe render와 schema/variant test 통과
  - 현재 Linux CUDA 8-GPU 및 verl runtime이 없어 `external-manual`,
    `result_origin=not_executed`, `local_executed=null`; 가짜 성능/성공 output 없음
- 미실행 범위: Qwen3-0.6B는 preflight only, QLoRA는 CPU에서 unsupported,
  MPS는 build=True/available=False인 sandbox probe 결과만 기록
- 검증: ruff pass, strict mypy 56 source files, pytest 117 passed in 97.51s,
  design contract, C8 JSON, dependency integrity와 `git diff --check` 통과
- 다음 작업: CalculatorToolEnv/LocalLookupEnv typed step contract, agent token mask,
  outcome/process reward와 credit assignment, multi-turn train/eval/failure case 구현

## 2026-08-19 — C9 Agentic RL 완료

- 현재 checkpoint: C10 한국어·영어 notebook과 Colab
- offline Agentic contract:
  - `CalculatorToolEnv`: `eval` 없는 AST arithmetic allowlist
  - `LocalLookupEnv`: immutable corpus와 answer+citation 채점
  - strict JSON action parser, visible tool schema, invalid/timeout,
    termination/truncation 분리; toy env에는 shell/network 실행 경로 없음
  - observation, 원 context/action token IDs, candidate set, parsed action, tool output,
    process/outcome reward와 policy version을 immutable step trajectory로 보존
  - 현재 action/EOS만 loss mask에 포함하고 tool output은 다음 context로만 사용
- policy/credit:
  - finite admissible-action LM scorer와 on-policy Agentic REINFORCE
  - final outcome broadcast baseline 대 process+outcome discounted return
  - outcome은 종료 step에만 두고 `process + outcome`을 한 번만 합산
- 실제 동일 조건 24-step CPU 비교:
  - shared initial policy `8c3c8b4…e4c5fe`, task split `2e24cdf…6fc0f4`,
    ordered episode IDs `cc4d5a7…14a299`, optimizer step 24
  - broadcast: env step 69, generated 2,094, processed 116,637,
    forward 162, validation success 0, useful-tool-step rate 0.125
  - discounted return: env step 67, generated 1,960, processed 110,253,
    forward 156, validation success 0, useful-tool-step rate 0.1818
  - 단일 seed·8 task이므로 우열을 주장하지 않음. 두 success 0과 broadcast
    last loss `-0.0`도 실패/한계 증거로 그대로 보존
- lifecycle/failure evidence:
  - continuous 4와 2+resume+2 model state가 모든 tensor `torch.equal`
  - CLI train/eval/inspect-run 성공, network socket deny train 성공
  - silent context truncation, stale/future trajectory, 허용 lag, tool nondeterminism,
    retokenization drift, repeated-call reward hacking, invalid/timeout/step-limit test
- ALFWorld 0.4.2 adapter:
  - injected backend와 최신 admissible command allowlist 경계 unit test 통과
  - package/assets 없는 Mac arm64에서는 download/install 없이 `external-manual`;
    ALFWorld MIT와 선택 Fast Downward GPL-3.0 경계를 분리
- 산출물: `docs/algorithms/agentic-rl.md`,
  `docs/research/C9_AGENTIC_BENCHMARK.json`, 두 Agentic config/experiment card
- 검증: ruff pass, strict mypy 65 source files, design contract, source/C9 JSON,
  `git diff --check` 통과; 전체 pytest 137 passed in 210.18s
- 다음 작업: 한국어 L00~L16 notebook을 고정 리듬/빠른·전체 path로 생성하고
  각 notebook을 clean top-to-bottom 실행한 뒤 영어 mirror/parity와 Colab 증거 완성

## 2026-08-19 — C10 진행 중: 한영 34개 검증 완료, hosted Colab 대기

- 한국어 L00~L16:
  - lesson별 핵심 식, 전체 지도상의 위치, 예측 질문, 작은 public-API 실험,
    구현 대안·trade-off, 구체적 함정/test, 회상 문제와 실제 출력 기반 결론 작성
  - L00 Mermaid와 ASCII fallback, 모든 lesson 3개 timed micro-section,
    `[필수/CORE]`/`[심화/DEEP DIVE]`, stable cell ID/source/test/code hash 적용
  - 17/17 independent fresh-kernel 실행 성공, 실행 contract 통과
- 영어 mirror L00~L16:
  - 한국어 최종 검증 뒤 동일 generator source에서 생성
  - 17/17 independent fresh-kernel 실행 성공
  - 17쌍 stable ID, code/hash, equation, source/test metadata parity 통과
- 최종 latest clean set:
  - 34/34 성공, Python 3.10.12, PyTorch 2.13.0, Darwin arm64
  - 한국어 코드 실행 합계 28.50초/peak RSS 528,318,464 byte
  - 영어 코드 실행 합계 26.89초/peak RSS 535,986,176 byte
  - append-only manifest 83 attempt 중 성공 71/실패 12; 실패 record를 삭제하지 않음
  - 산출물: `docs/research/C10_NOTEBOOK_EXECUTIONS.jsonl`,
    `docs/research/C10_NOTEBOOK_REPORT.json`
- 환경 진단:
  - repository `.venv`의 PyTorch Python file 7,279개가 macOS `dataless`로
    offload되어 cell import timeout이 발생함
  - 사용자 파일을 삭제하지 않고 `/private/tmp` 검증 venv에 동일 PyTorch 2.13.0과
    notebook dependency를 설치; `pip check` 통과 후 최종 증거 수집
  - 실패 timeout/permission/warning 기록도 manifest에 보존
- Colab source:
  - `notebooks/colab/RL_study_quickstart.ipynb`와 generator/contract checker 완성
  - clone → base install → offline foundation smoke + 2-action policy update →
    opt-in Qwen3-0.6B LoRA one-step 순서, download guard/tag/syntax/hash 검사 통과
  - 2026-08-19 `git ls-remote --symref origin HEAD`는 ref 없이 정상 종료:
    대상 원격이 비어 있어 hosted clone은 현재 실패함
  - 원격 push는 GOAL 계약상 별도 사용자 승인이 필요하므로 hosted 새 runtime
    실행과 날짜 증거는 아직 완료 처리하지 않음
- 다음 작업: 사용자가 첫 commit/push를 승인하면 populated remote를 재확인하고
  무료 Colab 새 CPU runtime에서 toy 경로를 실행해 C10을 완료

## 2026-08-19 — C11 로컬 완료: hosted 3-OS CI·scheduled job 대기

- 현재 checkpoint: C10/C11 hosted evidence gate, C12 local release audit 준비
- 실제 demo/report/UI:
  - DPO, RLHF-PPO, GRPO, DAPO, Agentic REINFORCE를 seed 42로 각각 실제
    1 update 실행하고 고유 run directory 생성
  - `summary.json`, 1800×1275 PNG, JavaScript 없는 정적 report, offline 대화형
    filter UI, checkpoint 5개와 experiment card 5개 생성
  - 같은 TinyReasoning fixed prompt에서 reward/sampled KL/entropy/clip fraction을
    진단하고 Agentic metric은 의미가 달라 그래프에서 분리
  - fresh venv 첫 실행 command 25.473초(목표 600초), peak RSS 595,099,648 byte;
    한 seed/한 update 결과로 순위나 논문 규모 재현을 주장하지 않음
  - `C11_DEMO_EVIDENCE.json`에서 모든 artifact SHA-256, PNG 구조,
    checkpoint manifest file hash와 card field를 재검증
- 공개 문서·접근성:
  - README 15분 quickstart/course entry/hardware/troubleshooting/contribution 경로
  - MkDocs home, getting started, 17-lesson map, 수학 도구함, algorithm cards,
    hardware matrix, troubleshooting, provenance와 기존 implementation notes 연결
  - 정적 report의 caption/alt/text conclusion, UI의 label/aria-live/noscript,
    `textContent` rendering과 local embedded JSON 계약
  - MkDocs 1.6.1 + Material 9.7.7 `build --strict` 0.44초 성공
- provenance/community/CI:
  - source 33개, 문서 사용 ID 32개, unknown 0, copied/adapted repository 0;
    GSM8K/TinyReasoning redistribution 조건 누락을 감사 중 발견해 보완
  - CONTRIBUTING, original CODE_OF_CONDUCT, SECURITY, SUPPORT, issue/PR template
  - GitHub Actions에 Linux/macOS/Windows × Python 3.10/3.12 CPU matrix와 weekly
    network/link/34-notebook fresh execution job 정의
  - official action release 확인 후 checkout/setup-python/upload-artifact major v7 사용
- 신규 사용자 journey:
  - 새 `/private/tmp/rl-study-c11-journey.SNXcFg/venv`에서 `.[dev]` install과
    `pip check`, CPU preflight, README demo, artifact audit 성공
  - `python -m rl_study.cli`가 아무 출력 없이 끝나는 entrypoint bug를 발견해
    `__main__` guard와 subprocess regression test를 추가한 뒤 fresh venv 재검증
  - local link 64 target 통과; network 44 URL 통과, pytest docs 1개는 HTTP 429라
    reachable-but-content-unvalidated로 명시; provenance/MkDocs strict 통과
  - 상세 실행 순서와 한계: `docs/research/C11_USER_JOURNEY.json`
- 최종 local 회귀(fresh venv): pytest 145 passed in 113.09s, ruff all pass,
  strict mypy 66 source files pass; notebook contract 34, bilingual parity 17,
  Colab source contract와 design contract 재통과
- 미완료 경계:
  - 원격은 여전히 ref가 없는 빈 저장소이고 push는 별도 사용자 승인이 필요함
  - 따라서 GitHub-hosted 3-OS matrix/weekly job과 fresh hosted Colab은 아직
    실행하지 않았으며 성공 badge·결과를 만들지 않음
- 다음 작업: C12 local hygiene/release 문서·clean clone audit를 완료한 뒤,
  사용자 push 승인 시 populated remote에서 hosted CI와 Colab evidence를 수집

## 2026-08-19 — C12 로컬 재현 감사 완료: hosted evidence 대기

- 현재 checkpoint: C10/C11 hosted evidence gate; C12 local release audit 통과
- release 준비:
  - package/CFF version `0.1.0`, CHANGELOG, 알려진 한계와 release contract 완성
  - secret pattern, file size, symlink, JSON finite 값, version/traceability/origin과
    공개 community file을 자동 검사
- clean-clone에서 발견하고 수정한 결함:
  - 첫 감사에서 `.gitignore`의 `models/`가 `src/rl_study/models/`까지 제외해 실제
    clone에 핵심 모듈 6개가 없는 문제를 발견
  - 패턴을 top-level `/models/`로 한정하고 source package를 추적한 뒤 전체 pytest
    146개, ruff, strict mypy를 재통과; 수정 commit `c29ad1c`
  - sandbox 안의 Jupyter ZMQ socket 생성은 34/34가 동일 PermissionError로 거부되어
    환경 제약으로 분리하고, 새 clone·새 venv의 비샌드박스 감사로 재검증
- 최종 clean-clone 감사:
  - 최초 `git status`가 빈 clone, branch `main`, commit
    `c29ad1c4967b9411a68dd847d73b3e3b2502a890`, exact GitHub origin 확인
  - macOS arm64, Python 3.10.12, PyTorch 2.13.0의 독립 venv에서 17/17 명령 통과,
    총 205.851초, auditor peak RSS 195,493,888 byte
  - pytest 146개, ruff, strict mypy 66 source, design/provenance/local-link/release,
    notebook 34개와 17쌍 parity, Colab source, strict MkDocs 모두 통과
  - 같은 감사에서 notebook 34/34를 각 fresh kernel로 재실행(52.737초)한 뒤
    executed/parity contract를 다시 통과
  - 명령별 반환 코드·시간·출력 SHA-256과 한계는
    `docs/research/C12_FINAL_AUDIT.json`에 원자적으로 기록
- 정직한 완료 경계:
  - local status는 `local_pass_hosted_pending`; 이 감사는 GitHub-hosted CI나
    fresh hosted Colab 성공을 뜻하지 않음
  - R19는 exact origin/clean clone/no-force 조건으로 verified. R11/R17/R20은 첫
    non-force push 후 hosted 실행 증거를 얻기 전까지 완료 처리하지 않음
- 다음 작업: 사용자 승인 후 빈 origin을 다시 읽기 전용 확인하고 `main`을
  non-force push한 뒤 3-OS CI/weekly job과 fresh Colab을 검증

## 2026-08-20 — C10~C12 hosted gate 완료

- 현재 checkpoint: C0~C12 완료
- 격리 환경:
  - local 최종 회귀는 `/private/tmp/rl-study-conda` Conda prefix의 Python 3.12 사용
  - base shell은 활성 상태였지만 base 환경에는 dependency를 설치하지 않음
  - `pip check`에서 broken requirement 0
- 최종 코드 후보:
  - commit `810205e55873c023a6d27accf9b72e9d1b7a9477`
  - `git push origin main`을 force 없이 실행
- GitHub-hosted CI:
  - run `32323026561`, conclusion `success`
  - Ubuntu/macOS/Windows × Python 3.10/3.12 CPU 6개와 static/docs job 모두 성공
  - Windows에서 발견한 POSIX 전용 peak-RSS/system-memory 경로를 Win32 API와
    cross-platform helper로 수정한 뒤 full matrix 재검증
- scheduled learning-artifact audit:
  - workflow_dispatch run `32323036240`, job `96288767091`, conclusion `success`
  - external/local link, 34개 fresh-kernel notebook, parity/output/Colab source,
    strict MkDocs와 artifact upload 모두 성공
  - artifact `9390477024`, 1,110,483 byte,
    SHA-256 `5f4d3a10…1fcb1b75`
- fresh Colab CPU runtime:
  - 같은 commit을 clone하고 Python 3.12.13 / PyTorch 2.13.0+cpu에서 실행
  - toy demo `completed`, `result_origin=local_executed`, fallback false,
    experiment card 5개, wall 56.607초
  - simple policy `p_good` 0.5 → 0.916
  - 선택형 실제 모델 smoke는 opt-in false로 정직하게 `skipped`
- local 최종 회귀:
  - pytest 148 passed, ruff pass, strict mypy 67 source files
  - design/provenance/notebook 34/parity 17/Colab/local links/MkDocs strict 통과
  - traceability `designed`/`pending` 0; R12 8-GPU verl만 근거 있는
    `external-manual`
- durable evidence:
  - `docs/research/C10_COLAB_EVIDENCE.json`
  - `docs/research/C12_HOSTED_AUDIT.json`
  - `scripts/check_hosted_evidence.py`
