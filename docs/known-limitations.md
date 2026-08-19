# 알려진 한계

이 페이지는 release 품질의 일부입니다. 결과가 좋아 보이는 부분보다 **어디까지
검증하지 않았는지**를 먼저 확인하세요.

## 과학적 해석

- TinyReasoning과 TinyCausalLM 결과는 교육용 sanity check이며 논문 규모 재현이
  아닙니다.
- 대부분의 비교 report는 한 seed와 짧은 budget입니다. 통계적 우월성, 일반화,
  production 성능을 입증하지 않습니다.
- DPO, online RLHF, group RL과 Agentic RL은 데이터와 action 의미가 다릅니다.
  서로 다른 metric을 한 순위표로 합치지 않습니다.
- reward model은 현재 toy에서 length/format shortcut을 보였습니다. 이 실패를
  제거하지 않고 진단 evidence로 보존합니다.

## 모델과 하드웨어

- 공개 LM 실제 검증은 macOS arm64 CPU의 SmolLM2-135M LoRA 2-step입니다. 다른
  CPU, MPS, CUDA와 Windows의 실제 LM 학습 성공을 뜻하지 않습니다.
- Qwen3-0.6B는 preflight/Colab opt-in 경로이고 이 laptop에서 학습하지 않았습니다.
- QLoRA는 CUDA와 bitsandbytes가 모두 검증될 때만 지원합니다. CPU/MPS에서
  LoRA/full fine-tuning으로 조용히 바꾸지 않습니다.
- verl 0.9.0 DAPO/GSPO server recipe는 schema와 command만 검증했습니다. 현재
  Linux CUDA 8-GPU 실행 결과는 `external-manual`입니다.

## Agentic RL

- core agent는 완전 오프라인 finite candidate-action scaffold입니다. 임의 shell,
  browser, network tool을 실행하는 production agent가 아닙니다.
- calculator와 local lookup은 credit/mask/termination 학습용 작은 환경입니다.
- ALFWorld adapter는 injected backend unit test만 통과했습니다. package/assets와
  simulator full runtime은 이 Mac에 설치하지 않아 `external-manual`입니다.

## Notebook, Colab, CI

- 34개 notebook은 macOS arm64 CPU와 독립 fresh kernel에서 실행했습니다.
- GitHub 원격이 비어 있고 첫 push가 승인되지 않아 hosted Linux/macOS/Windows
  Actions와 weekly job은 아직 실행되지 않았습니다.
- 같은 이유로 Colab notebook source contract만 검증됐고 새 hosted runtime
  output은 없습니다. Colab image와 무료 accelerator availability는 drift할 수
  있습니다.

## 최신성

- 문헌·repository·model/dataset 감사 기준일은 2026-08-19입니다.
- `최신`은 그 기준일의 audited source를 뜻하며 이후 논문을 자동 포함하지 않습니다.
- dependency는 호환 범위와 일부 exact adapter version을 사용합니다. 새 major가
  나와도 검증 없이 지원한다고 표시하지 않습니다.

## 범위 밖

Offline RL, model-based RL, continuous control, multi-agent RL, imitation learning은
LLM RL과의 관계를 설명하지만 각각의 대규모 benchmark를 구현하거나 재현하지
않습니다. 모든 RL 분야를 망라했다는 주장이 아닙니다.

검증 상태는 [traceability matrix](design/traceability.md), 실행 실패와 다음 시도는
repository의 `PROGRESS.md`와 `docs/research` evidence를 확인하세요.
