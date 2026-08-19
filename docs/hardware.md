# 하드웨어와 실행 프로필

가장 작은 `toy`에서 시작하고, 목적이 분명할 때만 `laptop`, `server`로 이동합니다.
요청한 device를 사용할 수 없으면 오류를 내며 자동 fallback하지 않습니다.

## 기능 표

| 경로 | macOS CPU | macOS MPS | Linux CPU | Linux CUDA | Windows CPU | Windows CUDA |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| toy core·notebook | 지원 | 명시 선택 가능 | 지원 | 명시 선택 가능 | CI 대상 | 명시 선택 가능 |
| toy demo report | 지원 | 미지원, CPU 고정 | 지원 | CPU 고정 | CI 대상 | CPU 고정 |
| SmolLM2 LoRA preset | 실제 CPU 검증 | preflight 필요 | 수동 검증 | 수동 검증 | 수동 검증 | 수동 검증 |
| QLoRA | 미지원 | 미지원 | 미지원 | CUDA+bitsandbytes 조건부 | 미지원 | CUDA+bitsandbytes 조건부 |
| verl server recipe | 미실행 | 미지원 | GPU 없음 | 8-GPU external-manual | 미지원 | 미지원 |

`지원`은 코드 경로와 로컬/CI 계약을 뜻하며 모든 장치에서 이미 hosted 실행됐다는
뜻은 아닙니다. 정확한 실행 evidence는 experiment card와 traceability 상태를
확인하세요.

## toy

- 외부 download 없음
- 기본 CPU
- TinyCausalLM, generated TinyReasoning, 오프라인 tool 환경
- 강의, unit test, train/resume/eval, report 생성

```bash
python -m rl_study.cli preflight --profile toy --device cpu --json
python -m rl_study.demo --profile toy --non-interactive --json
```

## laptop

- 공개 causal LM + LoRA
- 실행 전 model ID, exact revision, SPDX license, bytes, cache와 메모리 표시
- 100MB 초과 asset은 `--accept-download` 없이는 받지 않음
- 기본 preset은 SmolLM2-135M-Instruct, 품질 확장은 Qwen3-0.6B

다운로드 전에 정보만 확인하세요.

```bash
python -m rl_study.cli preflight \
  --profile laptop --model laptop-smoke --device cpu --json
```

세부 메모리 산식과 실제 실행 증거는
[Laptop·server 상세](profiles/laptop-server.md)에 있습니다.

## server

- verl 0.9.0에 고정된 DAPO/GSPO recipe
- CUDA multi-GPU, 실제 parquet와 운영 환경이 필요
- 이 저장소의 현재 local Mac 결과는 `external-manual`

```bash
python -m rl_study.cli render-server \
  --config configs/server/qwen3_4b_dapo_verl.yaml --json
```

실행하지 않은 throughput, reward, memory 수치를 만들지 않습니다.
