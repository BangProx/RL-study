# RL-study에 기여하기

작은 오류 수정부터 새 lesson 제안까지 환영합니다. 이 프로젝트는 학습자가 수식,
코드, 실행 결과를 서로 추적할 수 있는 상태를 가장 중요하게 봅니다.

## 시작하기

```bash
git clone https://github.com/BangProx/RL-study.git
cd RL-study
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev,notebooks,docs]'
```

Windows에서는 `.venv\Scripts\python`을 사용하세요.

## 변경 종류별 최소 검증

| 변경 | 최소 명령 |
|---|---|
| Python 코드 | `python -m pytest` + `python -m ruff check .` + `python -m mypy src` |
| 한국어 lesson | 해당 notebook fresh 실행 + notebook contract |
| 영어 mirror | 한영 parity + 한영 해당 notebook fresh 실행 |
| 문서 | `python scripts/check_links.py --local` + `python -m mkdocs build --strict` |
| 알고리즘 식/loss | hand-calculation unit test + 식↔함수↔source ID 문서 |

## 연구 구현 기여 계약

1. 논문, 저자 공식 코드 또는 framework 공식 문서를 source ID로 기록합니다.
2. 가져온 코드가 있다면 exact 파일·revision·license와 변경점을 기록합니다.
3. license가 불명확하면 복사하지 않고 clean-room 가능 여부부터 논의합니다.
4. tensor shape, mask, reduction, gradient ownership을 설명합니다.
5. 실행하지 않은 결과를 넣지 않고 실패와 한계를 보존합니다.
6. toy 성능으로 논문 규모 우월성을 주장하지 않습니다.

## Notebook 리듬

5~8분 micro-section, 예측, 짧은 실행, 왜/대안, 흔한 실수, 즉시 검사, 회상 문제를
지킵니다. 기존 metadata와 stable lesson ID를 임의로 바꾸지 마세요. 전체 계약은
`docs/design/notebook-style.md`에 있습니다.

## Pull request

- 한 PR에 하나의 설명 가능한 목적을 둡니다.
- 변경 이유, 사용자 영향, 실제 검증 명령과 결과를 적습니다.
- 생성 artifact, model weight, cache, secret, 대규모 dataset을 commit하지 않습니다.
- 기존 사용자 변경이나 실패 evidence를 삭제해 테스트를 통과시키지 않습니다.

질문은 issue의 lesson feedback template을 사용해 주세요. 보안 취약점은 공개
issue 대신 `SECURITY.md`의 절차를 따릅니다.
