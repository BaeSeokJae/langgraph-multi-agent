# LangGraph Multi-Agent System

PM, Dev, QA 에이전트를 체이닝한 LangGraph 기반 멀티 에이전트 시스템입니다.

## 아키텍처

```
User Task
    ↓
┌─────────────┐
│  PM Agent   │  ← 요구사항 분석 및 acceptance criteria 정의
└─────────────┘
    ↓
┌─────────────┐
│  Dev Agent  │  ← 코드 작성 (또는 QA 피드백 기반 수정)
└─────────────┘
    ↓
┌─────────────┐
│  QA Agent   │  ← 코드 테스트 및 검증
└─────────────┘
    ↓
  Pass? ──No──→ (Dev Agent로 재순환, max 3회)
    ↓
   Yes
    ↓
   END
```

## 주요 기능

### 1. 멀티 에이전트 워크플로우
- **PM Agent**: 요구사항 분석 및 acceptance criteria 정의
- **Dev Agent**: 코드 작성 및 QA 피드백 기반 수정
- **QA Agent**: 코드 테스트 및 검증

### 2. 슬래시 명령어

프로그램 실행 중 `/`로 시작하는 명령어로 설정을 변경할 수 있습니다:

| 명령어 | 설명 | 예시 |
|--------|------|------|
| `/help` | 명령어 도움말 표시 | `/help` |
| `/usage` | API 사용량 통계 표시 | `/usage` |
| `/config` | 현재 설정 표시 | `/config` |
| `/model` | 에이전트별 모델 설정 | `/model pm gpt-4` |
| `/temp` | 에이전트별 온도 설정 | `/temp dev 0.2` |
| `/max-iter` | 최대 반복 횟수 설정 | `/max-iter 5` |
| `/reset` | 사용량 통계 초기화 | `/reset` |

#### 명령어 예시

```bash
# PM 에이전트만 GPT-4 사용
/model pm gpt-4

# Dev 에이전트는 더 저렴한 모델 사용
/model dev gpt-3.5-turbo

# 모든 에이전트 모델 한번에 변경
/model all gpt-4o

# Dev 에이전트의 온도를 낮춰서 더 결정적으로
/temp dev 0.1

# 최대 반복 횟수 증가
/max-iter 5

# 사용량 확인
/usage
```

### 3. 실시간 진행 상황 표시
- 각 에이전트의 시작/완료 알림
- 코드 미리보기
- QA 결과 실시간 표시
- Rich 라이브러리 기반 색상 및 포맷팅

### 4. 한글 입력 지원
- `prompt_toolkit` 기반 입력 UI
- 백스페이스, 히스토리 등 완벽 지원

### 5. 연속 작업 처리
- 한 작업 완료 후 계속 새로운 작업 입력 가능
- 파일 저장 선택 사항

## 설치 및 실행

### 1. 환경 변수 설정

`.env` 파일을 생성하고 OpenAI API 키를 설정하세요:

```bash
cp .env.example .env
```

`.env` 파일 편집:
```bash
# 필수
OPENAI_API_KEY=your-api-key-here

# 선택사항 (에이전트별 다른 모델 사용 시)
LLM_MODEL=gpt-4
PM_MODEL=gpt-4
DEV_MODEL=gpt-4
QA_MODEL=gpt-3.5-turbo  # QA는 저렴한 모델 사용

# 선택사항 (에이전트별 온도 조절)
PM_TEMPERATURE=0.7
DEV_TEMPERATURE=0.3  # Dev는 더 결정적
QA_TEMPERATURE=0.2   # QA는 매우 결정적

# 워크플로우 설정
MAX_ITERATIONS=3
```

### 2. 의존성 설치

```bash
uv sync
```

### 3. 실행

```bash
uv run python main.py
```

또는 가상환경을 활성화한 후:

```bash
python main.py
```

## 사용 예시

```
============================================================
        Multi-Agent System: PM -> Dev -> QA
============================================================

💡 팁: 빈 입력, 'exit', 'quit'로 종료 | 슬래시 명령어는 /help 참고

작업 설명을 입력하세요 (Enter로 제출)
> 피보나치 수열을 계산하는 함수를 만들어줘

🚀 워크플로우를 시작합니다...

╭─────────────────────────────────────────────────╮
│ 📋 PM Agent | 요구사항 분석 중...              │
╰─────────────────────────────────────────────────╯
✅ PM 완료 | 요구사항 분석 완료

╭──────────── 📋 요구사항 분석 결과 ─────────────╮
│ 1. 피보나치 수열 계산 함수 구현               │
│ 2. n번째 항 반환                               │
│ ...                                             │
╰─────────────────────────────────────────────────╯

╭─────────────────────────────────────────────────╮
│ 💻 Dev Agent | 코드 작성 중...                 │
╰─────────────────────────────────────────────────╯
✅ Dev 완료 | 코드 생성 완료

╭──────────── 💻 생성된 코드 미리보기 ───────────╮
│  1  def fibonacci(n):                           │
│  2      if n <= 1:                              │
│  3          return n                            │
│  4      return fibonacci(n-1) + fibonacci(n-2)  │
╰─────────────────────────────────────────────────╯

╭─────────────────────────────────────────────────╮
│ 🧪 QA Agent | 코드 테스트 중...                │
╰─────────────────────────────────────────────────╯
✅ QA 완료 | 테스트 완료

╭──────────────── 🧪 QA 결과 ─────────────────────╮
│ ✅ 모든 테스트를 통과했습니다!                  │
╰─────────────────────────────────────────────────╯

코드를 파일로 저장하시겠습니까? [Y/n]
> y

파일명을 입력하세요 [기본값: output.py]
> fibonacci.py

✅ 코드가 fibonacci.py에 저장되었습니다.

다른 작업을 계속하시겠습니까? [Y/n]
> y

작업 설명을 입력하세요 (Enter로 제출)
> /usage

📊 사용량 통계
┌─────────────────┬────┐
│ 항목            │ 값 │
├─────────────────┼────┤
│ 총 API 요청     │  3 │
│ PM Agent 요청   │  1 │
│ Dev Agent 요청  │  1 │
│ QA Agent 요청   │  1 │
│                 │    │
│ 완료된 워크플로우│  1 │
│ 실패한 워크플로우│  0 │
└─────────────────┴────┘

작업 설명을 입력하세요 (Enter로 제출)
> exit

👋 프로그램을 종료합니다. 감사합니다!
```

## 프로젝트 구조

```
langgraph-practice/
├── agents/
│   ├── pm_agent.py          # PM 에이전트
│   ├── dev_agent.py         # 개발 에이전트
│   └── qa_agent.py          # QA 에이전트
├── tools/
│   ├── ui_utils.py          # UI 유틸리티 (Rich, prompt_toolkit)
│   └── __init__.py
├── state.py                 # 에이전트 간 공유 상태
├── graph.py                 # LangGraph 워크플로우
├── config.py                # 설정 관리 (동적 변경 가능)
├── commands.py              # 슬래시 명령어 핸들러
├── main.py                  # 진입점
└── .env                     # 환경 변수
```

## 비용 최적화 팁

에이전트별로 다른 모델을 사용하여 비용을 절감할 수 있습니다:

```bash
# PM과 Dev는 중요하니 GPT-4 사용
/model pm gpt-4
/model dev gpt-4

# QA는 단순 검증이니 저렴한 모델 사용
/model qa gpt-3.5-turbo
```

또는 `.env` 파일에서:
```bash
PM_MODEL=gpt-4
DEV_MODEL=gpt-4
QA_MODEL=gpt-3.5-turbo
```

## 확장 아이디어

`EXPANSION_IDEAS.md` 파일에서 확장 가능한 기능들을 확인하세요:
- Security Agent 추가
- Performance Agent 추가
- Documentation Agent 추가
- 외부 시스템 통합 (GitHub, Jira 등)
- 멀티모달 지원

## 기술 스택

- **LangGraph**: 멀티 에이전트 워크플로우 오케스트레이션
- **LangChain**: LLM 인터페이스
- **OpenAI GPT-4**: 각 에이전트의 LLM 백엔드
- **Rich**: 터미널 UI 및 포맷팅
- **prompt_toolkit**: 향상된 입력 처리
- **Python 3.13+**

## 라이선스

MIT
