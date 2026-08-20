# 출처, 라이선스, provenance

이 저장소의 기본 원칙은 **논문 식과 공개 동작을 근거로 새로 작성하고, 출처와
의도적 차이를 기록하며, 실행하지 않은 결과를 만들지 않는 것**입니다.

## 세 개의 원장

1. [`sources.yml`](sources.yml): 논문·공식 코드·framework·model·dataset의 exact
   metadata, revision, license, 확인일, 참고 지점
2. [C1 문헌 감사](research/C1_SOURCE_AUDIT.md): 구현 선택과 license 판단의
   사람이 읽는 근거
3. `experiment-card.json`: local 실행의 Git/config/environment/budget/result

C1~C9 실행 파일의 역할과 C3 번호가 비어 있는 이유는
[실행 근거 지도](research/README.md)에 정리했습니다.

## 재사용 분류

| 분류 | 의미 |
|---|---|
| `clean-room-reimplemented` | 수식·algorithm 설명과 public behavior를 보고 새 코드 작성 |
| `adapted` | 호환 license source를 수정했으며 원 파일·license·차이를 기록 |
| `copied` | 호환 license source를 그대로 포함하고 NOTICE/attribution 보존 |
| `architecture-reference` | 링크와 동작 비교에만 사용하고 코드를 가져오지 않음 |
| `optional-runtime-dependency` | 사용자가 별도 설치하며 이 저장소가 재배포하지 않음 |

현재 third-party source file을 vendor, copy, adapt하지 않았습니다. model weight,
dataset, 외부 benchmark와 framework도 저장소에 재배포하지 않습니다.

## DAPO의 특별한 경계

C1 감사 시점의 DAPO 공식 repository에는 명시 license가 없었습니다. 그래서
그 repository의 코드·주석·문서를 복사하거나 변형하지 않고, 논문 식만으로
`algorithms/dapo.py`를 clean-room 구현했습니다. license가 나중에 추가돼도 당시
판단과 출처 날짜를 소급해 바꾸지 않습니다.

## 결과 출처를 섞지 않기

- `paper_reported`: 논문이 보고한 값
- `upstream_reported`: 공식 구현이 보고한 값
- `local_executed`: 이 저장소에서 실제 실행한 값
- `external-manual`: 필요한 hardware가 없어 exact command/schema만 검증
- `not_executed`: 실행하지 않았고 local 수치는 `null`

toy 결과로 paper 결과를 재현했다고 쓰거나, failed run을 지우고 좋은 seed만
고르지 않습니다. append-only 실행 manifest에는 실패도 남깁니다.

## 생성된 notebook

`lessons/catalog.yml`이 한영 lesson의 구조적 원천입니다. 한국어 notebook을 먼저
작성·실행한 뒤 영어 mirror를 생성하며, parity checker가 objective, equation,
code hash, source ID와 section metadata를 비교합니다. 번역은 출처 license를
바꾸지 않습니다.

## Dependency와 asset

Python dependency는 각 upstream license를 유지합니다. Apache-2.0은 repository가
새로 작성한 코드·문서에 적용되며, dependency를 재라이선스하지 않습니다.
100MB 초과 model/dataset은 Git에 넣지 않고 명시 승인 후 upstream cache로 받습니다.

오류나 누락을 발견하면 [보안 정책](https://github.com/BangProx/RL-study/blob/main/SECURITY.md)에
해당하지 않는 provenance 문제는 일반 issue로 보고해 주세요.
