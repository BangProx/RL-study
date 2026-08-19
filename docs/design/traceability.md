# 요구사항 Traceability Matrix

> 상태 값: `designed`는 C2에서 산출물·검증 위치가 정해졌지만 아직 실행 증거가
> 없다는 뜻이다. `verified`는 실제 명령과 artifact가 있을 때만 사용하며,
> hardware가 없는 범위는 schema/command 검증 후 `external-manual`이 될 수 있다.

## 20개 완료 조건

| ID | 요구사항 요약 | Lesson/문서 | 구현 산출물 | 검증 증거 | 현재 상태 |
|---:|---|---|---|---|---|
| R01 | 한국어 17 + 영어 mirror 17 동등 | L00~L16, notebook style §5 | `notebooks/{ko,en}`, lesson spec generator | contract 34 + bilingual parity 17쌍 | verified |
| R02 | 모든 기본 notebook clean 실행 | notebook style §1, §11 | notebook executor | 34/34 fresh-kernel clean set + append-only C10 manifest | verified |
| R03 | 빠른 4~6h, 전체 11~14h | curriculum 빠른 경로 | notebook timing aggregator | README 369/845분 + 실제 코드 실행 28.50/26.89초 | verified |
| R04 | offline CPU toy가 bandit~DAPO/Agentic 실행 | L00~L16 | core env/model/algorithm, toy configs | offline integration suite + network-deny run | verified |
| R05 | 11개 필수 clean-room 구현 + CLI lifecycle | L04~L13 | `algorithms/*`, runner/CLI | analytic/unit + train/eval/checkpoint/resume | verified |
| R06 | classic PPO와 RLHF-PPO 차이 | L08~L10, glossary | `ppo.py`, `rlhf_ppo.py`, mask types | `rlhf-ppo.md`, `test_rlhf_ppo.py`, `test_sequence_masks.py` | verified |
| R07 | PPO/DPO/GRPO/DAPO 식↔코드·대안·ablation | L08, L11~L13 | algorithm cards + pure losses | source-link lint + analytic tests + ablation reports | verified |
| R08 | preference/verifier→update→eval 전체 pipeline | L10~L13 | TinyReasoning, RM/verifier, trainers | C6 fair comparison JSON + alignment lifecycle/card tests | verified |
| R09 | offline Agentic env 2 + external adapter | L15, C1 audit §7 | calculator/lookup + ALFWorld adapter | deterministic env/update tests; ALFWorld runtime은 external-manual | verified |
| R10 | toy/laptop/server와 선택형 config/CLI | L14, architecture §5 | strict config + adapter registry | unknown-key/profile matrix + CLI tests | verified |
| R11 | 공개 LM LoRA one-step + Colab | L14, Colab | TRL adapter, SmolLM2 preset, quickstart | local one-step/card와 Colab source contract 완료; hosted 새 runtime은 첫 push 대기 | designed |
| R12 | pinned 분산 framework recipe | L14, C1 audit §4 | verl v0.9.0 recipes | schema/render tests; GPU run external-manual if absent | external-manual |
| R13 | 10분 toy demo가 JSON/PNG/HTML/checkpoint/UI 생성 | L00/L16 | `rl_study.demo`, reporting, UI | fresh venv 25.47초 + C11 artifact/hash/UI audit | verified |
| R14 | 모든 결과 experiment card 추적 | architecture §12 | artifact writer/reader | required fields, base lock, RAM/VRAM, atomic write와 5-card audit | verified |
| R15 | 모든 수식·주장·선택의 source | 각 lesson Sources, C1 | `sources.yml`, source IDs, algorithm cards | provenance contract: 33 source / 32 documented ID / unknown 0 | verified |
| R16 | license/NOTICE/provenance 보존 | C1 audit §8 | LICENSE, NOTICE, provenance checker | copied/adapted 0, DAPO no-copy와 asset redistribution audit | verified |
| R17 | tests/lint/type/parity/link/3OS/scheduled CI | notebook style §10 | tests/scripts/workflows | local 145 tests/static/parity/link 통과; hosted 3OS/schedule는 첫 push 대기 | designed |
| R18 | README/MkDocs에서 사용자 경로 탐색 | curriculum + docs IA | README, MkDocs, hardware/troubleshooting | fresh venv journey, local/network links, strict MkDocs build | verified |
| R19 | 올바른 origin·원격 이력 보존·hygiene | GOAL C0/C12 | Git metadata/community files | status/remote/diff/no-force evidence | C0 verified; final audit pending |
| R20 | 모든 matrix row verified/external-manual | 이 문서 | evidence auditor | no designed/pending row at C12 | designed |

## 알고리즘→식→코드→test 계획

| 알고리즘 | source ID | public 구현 지점 | 필수 test/ablation ID |
|---|---|---|---|
| Q-learning | `sutton-barto-rl2` | `algorithms.tabular.q_learning_update` | `test_q_terminal_target`, `test_q_off_policy_target` |
| DQN | `dqn-2013` | `algorithms.dqn.dqn_loss` | `test_dqn_target_detached`, `ablation_dqn_target_sync` |
| REINFORCE | `sutton-barto-rl2` | `algorithms.reinforce.reinforce_loss` | `test_reinforce_sign`, `ablation_baseline_variance` |
| Actor-Critic | `sutton-barto-rl2` | `algorithms.actor_critic.actor_critic_loss` | `test_actor_advantage_detached` |
| GAE | `gae-2015` | `math.returns.generalized_advantage_estimate` | `test_gae_analytic`, `test_gae_lambda_boundaries` |
| PPO | `ppo-2017` | `algorithms.ppo.ppo_policy_loss` | `test_ppo_ratio_identity`, `test_ppo_clip_sign_cases` |
| Reward model | `learning-to-summarize-2020` | `algorithms.reward_model.pairwise_reward_loss` | `test_pairwise_loss_matches_bradley_terry_hand_value`, C5/C6 length shortcut report |
| RLHF-PPO | `instructgpt-2022` | `algorithms.rlhf_ppo.rlhf_ppo_loss` | `test_rlhf_reward_plus_token_kl_decomposition`, `test_rlhf_ppo_ratio_one_and_gradient_ownership` |
| DPO | `dpo-2023` | `algorithms.dpo.dpo_loss` | `test_dpo_loss_matches_hand_calculation`, `test_dpo_pair_order_and_label_smoothing` |
| GRPO | `deepseekmath-grpo-2024` | `algorithms.grpo.grpo_loss` | `test_group_relative_advantages_and_zero_variance_group`, `test_reduction_variants_expose_length_weighting` |
| RLOO | `rloo-2024` | `algorithms.grpo.rloo_advantages`, `rloo_sequence_loss` | `test_rloo_and_dr_grpo_hand_values`, `test_rloo_uses_full_sequence_as_one_action` |
| Dr. GRPO | `dr-grpo-2025` | `algorithms.grpo.dr_grpo_advantages`, `reduce_group_tokens` | `test_rloo_and_dr_grpo_hand_values`, `test_reduction_variants_expose_length_weighting` |
| DAPO | `dapo-2025` | `algorithms.dapo.dapo_loss` | `test_clip_higher_changes_positive_but_not_negative_advantage`, `test_dynamic_sampling_filters_all_correct_and_all_wrong_groups`, `test_token_level_toggle_changes_unequal_length_reduction`, `test_overlong_reward_shaping_boundaries` |
| GSPO | `gspo-2025` | `algorithms.grpo.gspo_sequence_loss` | `test_gspo_uses_one_geometric_mean_ratio_per_sequence` |
| Agentic policy update | `agent-lightning-2025`, `agent-r1-2025` | `agentic.credit`, `agentic.trajectory` | action mask/credit/failure unit test, two-env lifecycle test |

## Mask/gradient truth-table 계획

| token/model | attention | policy loss | reward/value | gradient |
|---|---:|---:|---:|---|
| prompt token | yes | no | optional value context | no direct policy action loss |
| model response/action | yes | yes | yes | current policy |
| EOS action | yes | yes | yes | current policy |
| padding | no | no | no | none |
| tool/environment output | yes as future context | no | process reward source only | none |
| old policy | n/a | stored detached log-prob | n/a | frozen |
| reference policy | n/a | KL/DPO comparison | n/a | frozen |
| reward model/verifier | n/a | reward source | yes | frozen during policy update |
| value model | n/a | no | return regression | value optimizer only |

prompt/response/EOS/padding 행은 `test_prompt_and_action_mask_truth_table`과
`test_variable_lengths_mask_padding_and_eos`, model 역할 행은 C5 reward-model 및
C6 DPO/RLHF gradient ownership test로 검증됐다. tool/environment 행은 C9의
`test_rollout_preserves_original_action_tokens_and_masks_tool_output`으로 검증됐다.

## C6 실행 증거

- DPO hand loss, pair ordering, response-only sequence 합과 label smoothing test
- RLHF reward+sampled-KL decomposition, masked return, ratio=1, gradient ownership test
- DPO와 RLHF-PPO 모두 continuous == interrupted+resume state (`rtol=0`, `atol=0`)
- reward-model RLHF와 verifier ablation을 포함한 실제 end-to-end train
- 동일 SFT SHA-256, 동일 ordered prompt ID hash, 동일 8 policy optimizer step 비교
- generated/processed token, model forward, wall time 차이를 숨기지 않은 C6 JSON
- CLI `train → resume → eval → inspect-run`, atomic checkpoint와 experiment card

## C7 실행 증거

- GRPO population group normalization, zero-variance/minimum/duplicate group test
- k3 reference KL clone-equality 수치 guard의 값과 gradient test
- RLOO sequence action, Dr. GRPO fixed denominator, GSPO geometric sequence ratio test
- DAPO Clip-Higher, Dynamic Sampling, global token loss, soft overlong shaping 독립 test
- GRPO/DAPO/RLOO/Dr. GRPO/GSPO 모두 continuous == interrupted+resume state
  (`rtol=0`, `atol=0`)와 CLI `train → resume → eval → inspect-run`
- 동일 SFT SHA-256과 non-dynamic prompt hash를 강제한 10-way local component report
- dynamic rollout 48회 대 non-dynamic 12회, rejected/exhausted/token/forward 비용 기록
- DAPO unlicensed upstream no-copy 경계와 paper-only clean-room 문서화

## C8 실행 증거

- exact model/dataset manifest, 100MB download guard, cache/no-network preflight test
- arbitrary model의 revision/license/expected bytes 누락과 remote code를 network 전에 거부
- CPU/MPS/CUDA 실제 tensor probe, 항별 memory estimate, QLoRA CUDA+bitsandbytes gate
- GSM8K `main/train`의 deterministic train/validation split과 official-test 차단 test
- SmolLM2-135M-Instruct exact revision CPU LoRA 2-step 실제 실행:
  460,800 / 134,975,808 parameter만 trainable, base gradient 없음
- continuous 2 == 1+resume+1 adapter safetensors byte parity,
  SHA-256 `0c56d05d…c0890f9`, offline eval과 checkpoint file integrity 통과
- TRL 1.10.0 algorithm mapping과 exact dependency lock; DAPO/GSPO는 억지 mapping 금지
- verl 0.9.0 DAPO/GSPO recipe schema와 critical override test 통과. Linux CUDA
  server 미보유로 `external-manual/not_executed/local_executed=null`, 성공 수치 없음

## C9 실행 증거

- strict JSON parser, AST calculator allowlist와 immutable local lookup을 포함한
  두 offline multi-turn environment; network socket deny 상태에서도 train 성공
- 원 action token ID, candidate set, policy version, parsed action, tool output,
  process/outcome reward와 termination을 immutable trajectory로 보존
- 현재 action/EOS만 policy loss에 포함하고 tool output은 다음 prompt context로만
  넣는 mask truth table test
- outcome broadcast와 discounted step return을 동일 initial policy/task IDs/seed/
  24 optimizer step에서 실제 비교하고 token/forward/environment 비용 개별 기록
- context overflow, stale/future/asynchronous lag, nondeterministic tool,
  retokenization drift, repeated-call reward hacking, invalid/timeout/truncation test
- continuous 4 == interrupted 2+resume 2 model state (`torch.equal`)와 CLI
  `train → eval → inspect-run` 실행
- ALFWorld 0.4.2 adapter의 admissible-command allowlist와 process/outcome 분리 test;
  package/assets 미설치 Mac arm64 실제 benchmark는 `external-manual`
- 단일 seed·8 validation task에서 두 방식 success 0을 그대로 보고하며 우월성
  주장을 하지 않음. 상세 수치: `docs/research/C9_AGENTIC_BENCHMARK.json`

## Checkpoint별 상태 전이

- C3~C9: 구현·unit/integration evidence를 각 R row에 추가
- C10: R01~R03, R11의 notebook/Colab evidence 추가
- C11: R13, R17, R18의 UI/site/CI evidence 추가
- C12: `designed`/`pending`가 0인지 자동 검사하고, 실제 hardware가 없는 항목만
  이유·명령·기대 artifact가 있는 `external-manual`로 허용

`external-manual`은 “아마 된다”가 아니다. exact dependency/version, 실행 명령,
필요 hardware, 예상 schema와 성공/실패 판정법이 모두 있어야 한다.
