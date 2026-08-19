# 15분 시작

목표는 설명을 읽기 전에 **실제 학습 artifact 한 세트**를 만드는 것입니다.
첫 설치에서 package 다운로드 시간은 네트워크 속도에 따라 15분을 넘을 수 있지만,
toy 학습 자체는 외부 모델이나 API를 사용하지 않습니다.

## 1. 준비

- Python 3.10, 3.11 또는 3.12
- Git
- 약 2GB의 설치 여유 공간
- CPU만으로 가능

=== "macOS / Linux"

    ```bash
    git clone https://github.com/BangProx/RL-study.git
    cd RL-study
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -e '.[dev]'
    ```

=== "Windows PowerShell"

    ```powershell
    git clone https://github.com/BangProx/RL-study.git
    Set-Location RL-study
    py -3.12 -m venv .venv
    .venv\Scripts\python -m pip install --upgrade pip
    .venv\Scripts\python -m pip install -e ".[dev]"
    ```

## 2. 환경을 먼저 확인

```bash
python -m rl_study.cli preflight --profile toy --device cpu --json
```

`status`가 `passed`, `resolved_device`가 `cpu`여야 합니다. 요청한 accelerator가
없을 때 조용히 CPU로 바꾸지 않습니다.

## 3. 실제 비교 데모 실행

```bash
python -m rl_study.demo \
  --profile toy \
  --non-interactive \
  --output-dir artifacts/demo \
  --json
```

데모는 DPO, RLHF-PPO, GRPO, DAPO, Agentic REINFORCE를 각각 실제 한 update
실행합니다. 매번 새 `artifacts/demo/demo-*` 디렉터리를 만들므로 기존 결과를
덮어쓰지 않습니다.

| artifact | 질문 |
|---|---|
| `summary.json` | 어떤 config·환경·실행값이었나? |
| `comparison.png` | 같은 TinyReasoning prompt의 verifier reward는 어땠나? |
| `report.html` | JavaScript 없이 표와 해석을 볼 수 있나? |
| `compare.html` | 계열·지표를 바꾸며 학습 전후 응답을 볼 수 있나? |
| `checkpoints/**` | 실제 update를 저장하고 다시 검사할 수 있나? |
| `experiment-card.json` | seed·Git·환경·시간·메모리·budget이 남았나? |

Agentic RL은 task와 action 의미가 다르므로 TinyReasoning reward 그래프에 섞지
않습니다. 빈 metric은 0이 아니라 `적용 불가`입니다.

## 4. 첫 notebook

```bash
python -m pip install -e '.[notebooks]'
jupyter lab notebooks/ko/L00_rl_map.ipynb
```

L00의 `[필수/CORE]` 셀만 실행해도 됩니다. 다음에는
[강좌 지도](course-map.md)에서 자신의 경로를 고르세요.

문제가 생기면 [문제 해결](troubleshooting.md), 실제 모델을 쓰고 싶다면
[하드웨어·프로필](hardware.md)로 이동합니다.
