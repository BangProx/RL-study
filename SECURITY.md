# Security policy

## Supported versions

아직 pre-release인 동안 보안 수정은 기본 branch의 최신 revision에만 제공합니다.
model weight, downloaded dataset, optional external framework의 취약점은 각 upstream
정책도 확인해야 합니다.

## 비공개 보고가 필요한 것

- arbitrary code/command execution
- download guard 또는 `trust_remote_code=false` 우회
- path traversal, checkpoint overwrite/integrity bypass
- credential, private data 또는 unsafe tool action 노출

GitHub repository의 **Security → Report a vulnerability** private reporting 기능을
사용해 주세요. 해당 기능이 보이지 않으면 maintainer profile의 비공개 연락 수단을
사용하고 공개 issue에 exploit, token, 개인 정보를 올리지 마세요.

재현 단계, 영향, affected revision, OS/Python/PyTorch와 최소 proof를 포함하되 실제
타인의 시스템이나 데이터를 공격하지 마세요. 수신 확인과 수정 일정은 maintainer
가용성에 따라 달라질 수 있으며, 아직 SLA를 약속하지 않습니다.

일반적인 설치 오류, 성능 문제, 문서 링크 오류는 security issue가 아니므로 일반
issue template을 사용해 주세요.
