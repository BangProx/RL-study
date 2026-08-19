# 문제 해결

오류를 숨기기보다 **어느 경계에서 실패했는지** 확인합니다.

## Python 또는 설치 버전이 맞지 않음

```bash
python --version
python -m pip --version
python -m pip check
```

Python 3.10~3.12가 필요합니다. `pip`만 호출하면 다른 interpreter에 설치될 수
있으므로 항상 `python -m pip`를 사용하세요.

## `torch` import가 매우 느림

사용하지 않는 external device plugin 탐색이 원인일 수 있습니다.

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c "import torch; print(torch.__version__)"
```

실제 CUDA/XPU plugin을 쓴다면 이 환경변수를 무조건 적용하지 말고 해당 backend
설치 문서를 확인하세요.

## `CUDA is not available` 또는 MPS 오류

이 저장소는 명시한 device를 조용히 CPU로 바꾸지 않습니다.

```bash
python -m rl_study.cli preflight --profile toy --device cpu --json
```

먼저 CPU로 학습 계약을 확인한 뒤, 설치한 PyTorch가 해당 accelerator build인지
확인하세요. `--allow-device-fallback`은 사용자가 그 의미를 알고 선택할 때만 씁니다.

## Matplotlib cache 권한 경고

report는 headless `Agg` backend를 사용합니다. home cache를 쓸 수 없는 CI/container라면:

```bash
mkdir -p .cache/matplotlib
MPLCONFIGDIR=.cache/matplotlib python -m rl_study.demo --profile toy --json
```

## checkpoint가 이미 있다고 실패

checkpoint writer는 실수로 결과를 덮어쓰지 않습니다. demo는 매번 고유한
`demo-*` 디렉터리를 만들고, 개별 trainer는 다른 `--output-root`를 사용하거나
기존 checkpoint에서 `--resume`해야 합니다.

## download 승인이 필요함

100MB를 넘는 model/dataset은 오류가 아니라 안전장치로 멈춥니다. preflight의
ID, revision, license, bytes, cache 경로를 읽고 동의할 때만
`--accept-download`를 추가하세요. `trust_remote_code`는 기본 false입니다.

## notebook kernel을 찾지 못함

```bash
python -m pip install -e '.[notebooks]'
python -m ipykernel install --user --name rl-study --display-name "RL-study"
python scripts/execute_notebooks.py --language ko --kernel-name rl-study
```

각 notebook은 새 kernel에서 단독 실행되어야 합니다. 앞 notebook의 변수를
복사해 해결하지 마세요.

## Windows 경로와 activation

PowerShell에서는 `.venv\Scripts\Activate.ps1`, `cmd.exe`에서는
`.venv\Scripts\activate.bat`를 사용합니다. activation 없이도
`.venv\Scripts\python -m pytest`처럼 interpreter를 직접 지정할 수 있습니다.

## 결과가 논문 수치와 다름

정상일 수 있습니다. toy model, 한두 update, deterministic generated data는
논문 규모 재현이 아닙니다. `experiment-card.json`의 `known_deviations`, seed,
budget과 `result_origin`을 먼저 비교하세요.

해결되지 않으면 재현 명령, OS/Python/PyTorch, config, 오류 전문, secret을 제거한
experiment card를 포함해 issue를 작성하세요.
