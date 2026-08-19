# Notebook 작성·검증 계약

> 이 문서는 L00~L16 한국어판, 영어 mirror와 Colab notebook의 생성·리뷰·CI
> 계약이다. 사람이 보기 좋은 것뿐 아니라 clean execution과 한영 parity를
> 기계적으로 검증할 수 있어야 한다.

## 1. 독립 실행 원칙

- 모든 notebook은 저장된 kernel state 없이 위에서 아래로 실행한다.
- 다른 lesson의 variable, local checkpoint나 working directory에 의존하지 않는다.
- 첫 code cell은 저장소 root를 탐지하고 version/seed/profile/device/network
  상태를 출력한다.
- 기본 경로는 `toy`, CPU, offline이다. network cell은 optional tag와 같은 개념을
  확인하는 deterministic toy fallback을 가진다.
- markdown에 적힌 결과는 해당 notebook에서 실제 실행된 output만 근거로 한다.
- 예외 traceback, debug dump, 거대한 tensor와 progress spam을 저장하지 않는다.

## 2. 고정 section 순서

각 notebook의 level-2 heading은 정확히 다음 순서다.

1. `## Goal`
2. `## Setup`
3. `## Steps`
4. `## Checks`
5. `## 내가 자주 틀리는 것` / `## Mistakes I Revisit`
6. `## 60초 요약` / `## 60-Second Recap`
7. `## Next Steps`
8. `## Sources`

`Steps` 안의 level-3 micro-section은 5~8분이며, 시작 줄에 다음 표시가 있다.

```text
⏱ 6분 · 2/6 section · [필수/CORE]
```

경로 label은 `[필수/CORE]`, `[심화/DEEP DIVE]`, `[서버/SERVER]` 세 개만 쓴다.
이모지와 색은 장식이며 의미는 텍스트 label만으로도 완전해야 한다.

## 3. 반복 학습 리듬

각 핵심 micro-section은 가능한 한 이 순서를 따른다.

1. **먼저 예측/Predict:** 10~30초 안에 고를 수 있는 방향·shape·부호 질문
2. **짧은 설명/Explain:** 한 문단, 작은 식 또는 작은 그림 하나
3. **실행/Run:** 보통 25줄 이하, 질문 하나에 답하는 code cell
4. **즉시 확인/Check:** assertion 또는 1~3분 회상 문제
5. **왜 이렇게 구현했나/Why:** 선택, 대안, 메모리·안정성 trade-off
6. **흔한 함정/Trap:** 틀린 코드 → 관찰할 증상 → 수정 → 잡아내는 test ID
7. **쉬어가기/Checkpoint:** 현재까지 확보된 output과 재개 위치 한 줄

정답은 먼저 시도할 공간 뒤의 `<details><summary>정답 보기</summary>`에 둔다.
`Checks`에는 최소 한 개의 실행 가능한 assertion과 사람이 설명하는 회상 문제가
모두 있어야 한다.

## 4. Notebook metadata

notebook root metadata의 `rl_study` namespace는 다음 schema를 따른다.

```json
{
  "rl_study": {
    "schema_version": 1,
    "lesson_id": "L08",
    "language": "ko",
    "mirror_language": "en",
    "title_key": "actor_critic_gae_ppo",
    "profile": "toy",
    "estimated_minutes_full": 60,
    "estimated_minutes_fast": 32,
    "prerequisites": ["L05", "L07"],
    "source_ids": ["gae-2015", "ppo-2017", "repo-spinningup"],
    "network_required": false,
    "seed": 42,
    "generated_from": "lessons/L08.yml"
  }
}
```

필수 cell metadata:

```json
{
  "id": "l08-s03-ppo-ratio",
  "metadata": {
    "rl_study": {
      "stable_id": "L08.S03.C04",
      "kind": "code",
      "path": "core",
      "concept_ids": ["ppo.ratio", "ppo.old_policy"],
      "source_ids": ["ppo-2017"],
      "test_ids": ["test_ppo_ratio_identity"],
      "code_hash": "sha256:..."
    },
    "tags": ["rl-study-core"]
  }
}
```

- nbformat cell `id`는 소문자 ASCII slug이고 번역판에서 동일하다.
- `stable_id`는 `L{lesson}.S{section}.C{cell}`이며 삭제 후 다른 내용에 재사용하지
  않는다.
- code cell에는 `code_hash`가 필수다. normalized source는 LF, trailing whitespace
  제거, 마지막 newline 하나로 만든 뒤 SHA-256을 계산한다.
- markdown cell도 `stable_id`, `kind`, `path`, `concept_ids`, `source_ids`를 가진다.
- figure 생성 cell은 `figure_id`, 직후 markdown cell은 같은 `figure_id`와
  `alt_text`를 가진다.

## 5. 한영 mirror

한국어 17개가 contract와 clean execution을 모두 통과한 뒤 영어판을 생성한다.
한영판에서 같아야 하는 것:

- lesson/section/cell stable ID와 순서
- code source와 code hash
- 수식, seed, config, dataset IDs, source IDs
- exercise/check 수와 assertion
- figure data/hash, axis/marker/line style
- 저장한 실행 manifest의 schema

달라도 되는 것:

- 자연스러운 markdown 설명과 title
- figure의 사람이 읽는 label/alt text
- 한국어 `내가 자주 틀리는 것`과 영어 `Mistakes I Revisit` heading text

번역 때문에 수식의 가정, 난이도, 결론을 바꾸지 않는다. code comment는
language-neutral English를 canonical로 하거나 실행에 영향 없는 별도 markdown으로
설명한다.

## 6. Package와 code cell

package가 canonical 구현이다. notebook은 다음 두 형식 중 하나만 사용한다.

- 수식을 10~25줄로 재구성한 뒤 package 함수와 `allclose` parity를 확인한다.
- package의 public API를 호출하고 입력/출력 tensor의 값·shape·gradient를
  작은 표로 관찰한다.

trainer 전체, config loader, checkpoint writer를 notebook에 복사하지 않는다.
source view가 필요하면 고정된 public function 이름과 로컬 Git 링크를 보여주고,
line number는 CI에서 생성한다.

## 7. 출력·그림 접근성

- table은 기본 20행 이하이며 생략한 행 수를 표시한다.
- tensor는 shape, dtype, device와 관찰할 3~8개 값만 보인다.
- 모든 plot은 title, axis label, legend 또는 direct label을 가진다.
- 알고리즘 구분은 색 + marker + line style을 함께 사용한다.
- figure 직후 alt text와 “한 줄 결론”을 markdown으로 둔다.
- animation과 자동 재생 media는 사용하지 않는다.
- warning을 무시하지 않는다. 예상 warning은 원인을 설명하고 좁은 scope에서만
  filter한다.

권장 colorblind-safe palette는 Okabe–Ito 계열이지만, 색 자체는 의미 계약이
아니다.

## 8. Setup 출력

`Setup`은 secret이나 전체 environment dump 없이 다음을 한 번 출력한다.

```text
lesson=L08 language=ko profile=toy
seed=42 network_required=False
python=... rl_study=... torch=...
requested_device=cpu resolved_device=cpu fallback_used=False
config_hash=... data_split_hash=...
```

요청 device가 없으면 조용히 CPU로 바꾸지 않는다. `allow_device_fallback=true`가
명시된 경우에만 바꾸고 `fallback_used=True`를 출력한다.

## 9. Cell tag

| tag | 의미 | CI 동작 |
|---|---|---|
| `rl-study-core` | 기본 toy 경로 | 매 실행 포함 |
| `rl-study-deep-dive` | 심화지만 offline | full notebook 실행 포함 |
| `rl-study-network` | 명시적 다운로드 필요 | 기본 skip, scheduled network job에서만 실행 |
| `rl-study-server` | CUDA/분산 필요 | 문법·schema 검사, 실제 실행은 external-manual |
| `rl-study-slow` | 60초 이상 예상 | scheduled job에서 실행 |
| `rl-study-hide-solution` | 접힌 해설 | 렌더링 시 `<details>` 유지 |

network/server cell은 실행하지 않은 output을 repository에 미리 저장해 성공처럼
보이지 않게 한다. 바로 앞 core cell에서 같은 개념의 toy fallback을 실행한다.

## 10. Contract checker

`scripts/check_notebook_contract.py`가 최소 다음을 실패시킨다.

- 잘못된 section 순서 또는 목표 0개/4개 이상
- 중복·누락 stable ID, 잘못된 lesson/language metadata
- predict/check/mistake/recap/source cell 누락
- `Checks`에 실행 assertion 없음
- source ID가 `docs/sources.yml`에 없음
- network/server tag가 있는데 fallback이나 위험 설명이 없음
- figure alt text·텍스트 결론 누락
- 100KB가 넘는 단일 output, traceback, progress spam
- 절대 local path, secret 모양 문자열, 실행 번호가 뒤섞인 상태

`scripts/check_bilingual_parity.py`는 stable IDs, code hashes, equations, checks,
figures와 source IDs를 비교한다. 번역 markdown의 byte equality는 요구하지 않는다.

## 11. Clean execution manifest

각 실행은 아래 필드를 가진 JSONL record를 남긴다.

```json
{
  "lesson_id": "L08",
  "language": "ko",
  "profile": "toy",
  "started_at": "...",
  "duration_seconds": 12.3,
  "peak_rss_bytes": 123456789,
  "python": "3.10.12",
  "torch": "2.13.0",
  "platform": "macOS-arm64",
  "git_commit": "...",
  "dirty": false,
  "notebook_sha256": "...",
  "success": true,
  "error": null
}
```

manifest는 C10에서 생성하며, 실패한 record를 삭제해 성공률을 꾸미지 않는다.
