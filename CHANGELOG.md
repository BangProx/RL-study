# Changelog

이 프로젝트는 [Semantic Versioning](https://semver.org/)을 따릅니다. 날짜는 UTC
기준이며, GitHub release가 실제로 게시되기 전에는 이 문서의 항목도 로컬 release
준비 기록입니다.

## [0.1.0] - 2026-08-19

첫 공개 release 후보입니다. 아직 원격 push와 GitHub release는 수행하지 않았습니다.

### Added

- 17개 한국어 notebook과 구조·수식·코드가 같은 17개 영어 mirror
- 고전 RL부터 RLHF-PPO, DPO, GRPO, DAPO, RLOO, Dr. GRPO, GSPO까지의
  clean-room PyTorch reference 구현
- 두 오프라인 multi-turn tool 환경, Agentic REINFORCE와 ALFWorld adapter 경계
- `toy`, 공개 LM LoRA `laptop`, pinned verl `server` profile
- train/eval/checkpoint/resume CLI와 seed/config/environment/budget experiment card
- 5개 알고리즘 실제 one-step JSON/PNG/정적 HTML/대화형 UI demo
- ADHD 친화적 micro-section, 빠른/전체 학습 경로, MkDocs 사이트
- 한영 parity, fresh notebook execution, provenance, local/network link 검사
- Linux/macOS/Windows CPU CI와 weekly network/notebook audit workflow 정의

### Validation

- 한국어·영어 notebook 34/34 fresh-kernel clean execution
- 신규 virtualenv 전체 test 145 passed, ruff와 strict mypy 통과
- MkDocs strict build, local 64 targets와 external 44 URLs 검사
- SmolLM2-135M-Instruct CPU LoRA 실제 2-step과 resume byte parity

### Hosted validation

- Linux/macOS/Windows × Python 3.10/3.12 hosted CI, scheduled notebook audit와
  새 Colab CPU runtime toy 경로를 실행하고 durable evidence를 기록했습니다.
- Linux CUDA 8-GPU verl recipe와 ALFWorld full runtime은 `external-manual`입니다.

[0.1.0]: https://github.com/BangProx/RL-study/releases/tag/v0.1.0
