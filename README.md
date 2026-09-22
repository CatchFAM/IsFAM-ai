# IsFAM

IsFAM은 가족 사칭형 AI 보이스피싱을 줄이기 위한 음성 기반 가족 확인 시스템입니다.

핵심 목표는 "이 음성이 딥보이스인가?" 하나만 맞히는 것이 아니라, 전화 중 들리는 목소리를 **등록된 가족으로 신뢰해도 되는지**를 판단하는 것입니다. 이를 위해 IsFAM은 가족 화자 인증, 딥보이스 탐지, 통화 구간별 누적 판단을 결합한 하이브리드 AI 위험도 알고리즘을 사용합니다.

## 개발 요약

IsFAM에서 실제로 개발한 핵심 기능은 다음과 같습니다.

```text
1. 가족 음성 등록
   가족의 실제 음성을 AI speaker embedding으로 변환하고 DB에 voiceprint로 저장합니다.

2. 다중 voiceprint 기반 가족 프로필
   가족 1명당 여러 음성 샘플을 등록하고, 같은 name/relation을 하나의 가족 프로필로 묶습니다.

3. 통화 음성 유사도 계산
   새 통화 음성을 embedding으로 변환한 뒤, 등록 가족 voiceprint와 cosine similarity를 계산합니다.

4. median similarity 기반 안정화
   최고점 하나에 의존하지 않고 max/mean/median similarity를 함께 계산해 튀는 점수의 영향을 줄입니다.

5. 가족별 threshold 자동 산출
   등록된 가족 voiceprint끼리의 같은 화자/다른 화자 분포를 비교해 가족 프로필별 기준값을 자동 계산합니다.

6. 샘플 품질 가중치 적용
   margin이 좁은 등록 샘플은 판단 가중치를 낮춰 품질이 좋은 voiceprint가 최종 유사도에 더 크게 반영되게 합니다.

7. 가족 불일치 우선 탐지
   등록 가족과 충분히 비슷하지 않으면 AI 여부와 관계없이 가족 사칭 위험으로 판단합니다.

8. anti-spoofing 보조 판단
   딥보이스 탐지 모델의 spoof_score를 보조 신호로 사용해 위험도를 강화합니다.

9. 설명 가능한 위험도 출력
   safe/caution/danger, risk_score, family_confidence, mismatch_confidence, final_decision, decision_reasons를 API 응답으로 제공합니다.

10. 통화 chunk confidence 누적
   통화 구간별 family confidence와 mismatch confidence를 품질/최근성 가중치로 누적해 실시간 판단을 안정화합니다.

11. 객관 평가 리포트
   가족 음성과 공개 real/fake 고정 평가셋으로 정확도, FAR/FRR, fake recall, 지연 시간을 산출합니다.
```

현재 구현된 판단 흐름은 다음과 같습니다.

```text
가족 등록 음성
  -> speaker embedding 추출
  -> SQLite DB에 voiceprint 저장

새 통화 음성
  -> speaker embedding 추출
  -> 등록 가족 voiceprint와 similarity 계산
  -> 가족별 threshold와 샘플 품질 가중치 적용
  -> anti-spoofing spoof_score 계산
  -> IsFAM Risk Scoring
  -> safe / caution / danger
```

## 문제 정의

최근 보이스피싱은 단순히 낯선 번호로 돈을 요구하는 수준을 넘어, 가족의 목소리를 AI로 흉내 내는 방식으로 발전하고 있습니다. 사용자는 짧은 통화 안에서 상대가 진짜 가족인지, AI로 변조된 목소리인지 판단해야 합니다.

IsFAM은 다음 상황을 판별합니다.

```text
가족 실제 음성          -> 안전
가족이 아닌 실제 음성   -> 확인 필요 또는 위험
가족을 흉내 낸 AI 음성  -> 위험
일반 AI/TTS 음성        -> 위험
잡음이 심한 통화 음성   -> 확인 필요
```

## AI 구조

IsFAM은 단일 모델에 의존하지 않습니다. 현재 대회 시연 버전은 별도 가짜 가족 음성 데이터가 없어도 동작하도록 **가족 voiceprint 불일치 탐지**를 1차 기준으로 사용합니다.

```text
전화 음성
  -> 음성 전처리
  -> 가족 voiceprint 비교
  -> 딥보이스 탐지
  -> 가족 중심 IsFAM Risk Scoring
  -> 안전 / 확인 필요 / 위험
```

### 1. 가족 화자 인증

등록된 가족 음성을 speaker embedding으로 변환해 SQLite DB에 저장합니다. 전화 중 들어온 음성도 같은 방식으로 embedding을 추출한 뒤, 등록된 가족 voiceprint와 cosine similarity를 계산합니다.

사용 모델:

```text
speechbrain/spkrec-ecapa-voxceleb
```

### 2. 딥보이스 탐지

3~5초 전화 음성에서 log-spectrum 통계 특징 196개를 추출하고, 작은 MLP로 real/spoof 점수를 계산합니다. 공개 DFADD calibration 세트로 학습·보정했으며 기존 Transformers 모델은 선택 가능한 fallback으로 남겨 두었습니다.

기본 모델:

```text
isfam/spectral-mlp-dfadd-v1
```

고정 DFADD 테스트 800개에서 기존 기본 모델 대비 정확도는 `50.88% → 98.88%`, fake recall은 `1.75% → 99.50%`, 평균 순수 추론은 `117.18 ms → 2.08 ms`로 개선됐습니다. 모델 구성, 개선 이유, 전화환경 제한과 상세 수치는 [AI 최종 정리](reports/ai_upgrade_results.md)를 참고합니다.

### 3. IsFAM Risk Scoring

기존 오픈소스 모델의 출력값을 그대로 사용하지 않고, 팀만의 위험도 알고리즘으로 최종 판단을 만듭니다. 이때 가장 중요한 신호는 "등록된 가족 목소리와 충분히 비슷한가"입니다. 실제 보이스피싱에서는 AI 음성이 아니더라도 가족이 아닌 사람이 가족처럼 말하면 위험하기 때문입니다.

판단에 사용하는 신호는 다음과 같습니다.

```text
가족 voiceprint 유사도
등록 가족 기준 미달 여부
AI 합성 음성 의심 점수
등록 샘플 수
등록 샘플 간 점수 안정성
통화 chunk별 반복 경고 여부
rolling_family_confidence
rolling_mismatch_confidence
```

최종 출력은 다음 3단계입니다.

```text
safe     안전
caution  확인 필요
danger   위험
```

## 독창적 최적화

대회 기준에서 중요한 부분은 단순히 오픈소스 모델을 호출하는 것이 아니라, 서비스 문제에 맞게 AI 출력을 재해석하고 안정화하는 것입니다. IsFAM은 다음 최적화를 적용했습니다.

### 다중 voiceprint 집계

가족 1명당 음성 샘플을 여러 개 등록할 수 있습니다. 같은 `name`과 `relation`으로 등록된 voiceprint는 하나의 가족 프로필로 묶입니다.

단일 최고점만 사용하면 우연히 점수가 높게 나온 샘플 때문에 오판할 수 있으므로, IsFAM은 다음 값을 함께 계산합니다.

```text
sample_count
max_similarity
mean_similarity
median_similarity
```

최종 가족 매칭 점수는 안정성을 위해 median similarity를 중심으로 판단합니다.

### 가족 중심 위험도 정책

현재 IsFAM은 다음 기준으로 1차 탐지를 강화합니다.

```text
등록 가족 similarity >= 기준값
-> 안전 후보

등록 가족 similarity가 기준값에 근접
-> 확인 필요

등록 가족 similarity가 기준값보다 크게 낮음
-> AI 여부와 무관하게 위험
```

딥보이스 탐지 모델은 이 판단을 보조합니다.

```text
가족 아님 + AI 음성 의심 낮음
-> 등록 가족과 다른 목소리로 위험 또는 확인 필요

가족 아님 + AI 음성 의심 높음
-> 가족 사칭형 AI 음성 가능성으로 고위험

가족과 비슷함 + AI 음성 의심 높음
-> 가족 목소리 흉내 가능성으로 확인 필요 또는 위험
```

### 설명 가능한 위험도

`/api/v1/voice/verify` 응답은 단순 boolean만 반환하지 않습니다.

```json
{
  "is_trusted": false,
  "risk_level": "danger",
  "risk_score": 0.82,
  "final_decision": "spoofed_family_like_voice",
  "decision_reasons": [
    "가장 가까운 가족 voiceprint 유사도 0.6410가 기준에 근접하지만 통과하지 못했습니다.",
    "AI 합성 음성 의심 점수 0.4200가 강한 경고 기준에 도달했습니다."
  ]
}
```

이 구조는 심사위원과 사용자에게 AI가 왜 그런 판단을 했는지 설명할 수 있게 합니다.

### 통화 구간 누적 분석

실제 통화는 한 번의 음성 파일보다 불안정합니다. IsFAM은 voice session API를 통해 3~5초 단위 chunk를 누적 분석할 수 있습니다.

누적 판단에는 다음 정보가 사용됩니다.

```text
분석 가능한 chunk 수
저품질로 제외된 chunk 수
가족으로 확인된 chunk 수
spoof 의심 chunk 수
누적 가족 신뢰도 rolling_family_confidence
누적 불일치 신뢰도 rolling_mismatch_confidence
신뢰 chunk 수 trusted_chunks
불일치 chunk 수 mismatch_chunks
최대 spoof score
동일 가족이 반복 확인됐는지 여부
```

이를 통해 짧은 구간 하나의 실수보다 전체 통화 흐름을 기준으로 판단합니다. 최근 chunk와 음성 품질이 좋은 chunk에는 더 높은 가중치를 주고, 가족 불일치 confidence가 반복 누적되면 anti-spoofing 점수가 낮아도 위험 상태로 전환합니다.

## 평가 기준 대응

### AI 모델/알고리즘 적절성

IsFAM은 문제를 두 개의 AI 세부 문제로 분해합니다.

```text
가족인지 확인        -> Speaker Verification
AI 합성 음성인지 확인 -> Anti-Spoofing
최종 서비스 판단      -> IsFAM Risk Scoring
```

가족 사칭 보이스피싱은 "딥보이스 여부"만으로는 해결되지 않습니다. 진짜 사람 목소리라도 가족이 아니면 위험할 수 있기 때문에, 가족 인증과 딥보이스 탐지를 함께 사용합니다.

### 기존 오픈소스 활용을 넘는 최적화

적용된 팀 최적화는 다음과 같습니다.

```text
다중 가족 voiceprint 집계
median similarity 기반 가족 판단
가족별 threshold 자동 산출
품질 낮은 등록 샘플 가중치 완화
등록 가족 불일치 우선 탐지
AI 합성 점수의 보조 위험도 반영
등록 샘플 수에 따른 신뢰도 조정
family_confidence / mismatch_confidence 산출
chunk 기반 누적 위험 판단
통화 chunk별 family/mismatch confidence 누적
판단 근거 문장 생성
```

### 입력값에 대한 안정적 출력

결과는 단순히 `true/false`가 아니라 다음처럼 안정적인 상태로 제공됩니다.

```text
safe     가족으로 신뢰 가능
caution  가족 확인 필요
danger   보이스피싱 또는 AI 사칭 의심
```

애매한 입력은 억지로 안전 처리하지 않고 `caution`으로 분리합니다.

### 기술 지표

대회 시연 및 평가 데이터에서 다음 지표를 측정할 수 있습니다.

```text
가족/비가족 판별 정확도
AI 합성 음성 탐지 정확도
위험 통화 탐지율
정상 가족 통화 오탐률
평균 처리 시간
chunk 1개당 분석 시간
```

권장 목표:

```text
가족/비가족 판별 정확도 85% 이상
위험 통화 탐지율 90% 이상
chunk 분석 시간 2초 이하
```

현재 등록 가족 음성 10개 기준으로 speaker verification 분리 성능을 측정했습니다.

```text
파일 수: 10개
화자 수: 2명(mom, daughter)
같은 화자 pair: 20개
다른 화자 pair: 25개
같은 화자 평균 유사도: 0.7898
같은 화자 최소 유사도: 0.6676
다른 화자 평균 유사도: 0.4311
다른 화자 최대 유사도: 0.5021
분리 여유도(min same - max different): 0.1655
threshold 0.65 기준 pair accuracy: 100%
threshold 0.65 기준 false accept rate: 0%
threshold 0.65 기준 false reject rate: 0%
leave-one-out 화자 식별 정확도: 100% (10/10)
추천 threshold: 0.65
신뢰도 등급: high
```

현재 데이터에서는 `ISFAM_SPEAKER_THRESHOLD=0.65`가 가장 안정적입니다. 추천 기준은 false accept가 0%인 후보 중 정확도가 높고, 가족 통과 기준을 가장 엄격하게 유지하는 threshold를 선택합니다.

샘플 품질 자동 점검 결과, 다음 2개 파일은 같은 화자 최소 유사도가 threshold에 가까워 추후 재녹음하면 안정성이 더 좋아집니다.

```text
mom_register_03.m4a
mom_register_05.m4a
```

단, 현재 데이터에서는 이 두 파일을 포함해도 threshold 0.65 기준 정확도, false accept rate, false reject rate가 모두 목표 기준을 만족합니다. 이 지표는 등록 가족 음성 내부 분리 성능입니다. 가짜 가족 음성이나 실제 비가족 음성이 없기 때문에 딥보이스 탐지 정확도와 외부 비가족 탐지 정확도는 별도 데이터 확보 전까지 정량화할 수 없습니다.

#### 지표 읽는 법

유사도는 높을수록 등록 가족 목소리와 비슷하다는 의미입니다.

```text
similarity 높음 -> 같은 사람일 가능성이 높음
similarity 낮음 -> 다른 사람일 가능성이 높음
```

현재 데이터에서 중요한 값은 다음 두 개입니다.

```text
같은 화자 최소 유사도: 0.6676
다른 화자 최대 유사도: 0.5021
```

같은 사람끼리 비교한 최저점이 `0.6676`이고, 다른 사람끼리 비교한 최고점이 `0.5021`입니다. 따라서 기준값을 `0.65`로 잡으면 현재 데이터에서는 같은 사람은 통과하고 다른 사람은 거절됩니다.

```text
0.65 이상 -> 등록 가족으로 판단
0.65 미만 -> 등록 가족과 다른 목소리로 판단
```

`분리 여유도 0.1655`는 두 분포 사이의 빈 공간입니다.

```text
0.6676 - 0.5021 = 0.1655
```

이 값이 클수록 같은 가족 목소리와 다른 목소리가 더 잘 분리됩니다.

### 데이터 수집/처리 타당성

모델을 새로 학습하지 않더라도 평가 데이터는 필요합니다. IsFAM은 다음 데이터셋 구조를 기준으로 threshold와 위험도 점수를 검증합니다.

```text
datasets/eval/
  family_real/        등록 가족의 실제 음성
  non_family_real/    가족이 아닌 사람의 실제 음성
  family_deepvoice/   가족 목소리를 AI로 흉내 낸 음성
  ai_voice/           일반 TTS 또는 AI 음성
  noisy_call/         잡음이 포함된 통화 음성
```

이 데이터는 모델 학습용이 아니라, 서비스 판단 기준 검증과 성능 측정용입니다.

## 주요 API

### 가족 음성 등록

```bash
curl -X POST http://127.0.0.1:8000/api/v1/family/register \
  -F "name=엄마" \
  -F "relation=mother" \
  -F "audio_file=@samples/mother_01.wav"
```

같은 가족을 여러 번 등록할 때는 같은 `name`, `relation`을 사용합니다.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/family/register \
  -F "name=엄마" \
  -F "relation=mother" \
  -F "audio_file=@samples/mother_02.wav"
```

### 가족 여부 확인

```bash
curl -X POST http://127.0.0.1:8000/api/v1/voice/verify-family \
  -F "audio_file=@samples/call.wav"
```

### 가족 여부 + 딥보이스 통합 판단

```bash
curl -X POST http://127.0.0.1:8000/api/v1/voice/verify \
  -F "audio_file=@samples/call.wav"
```

응답 예시:

```json
{
  "is_trusted": true,
  "risk_level": "safe",
  "risk_score": 0.04,
  "final_decision": "trusted_family_voice",
  "processing_time_ms": 842.31,
  "family_model_time_ms": 231.45,
  "anti_spoofing_model_time_ms": 610.52,
  "decision_reasons": [
    "엄마 voiceprint와 유사도 0.8123로 기준보다 0.0623 높습니다.",
    "AI 합성 음성 의심 점수 0.0100가 기준보다 낮습니다.",
    "3개 등록 샘플 기준 유사도 편차가 안정적입니다."
  ]
}
```

세 시간 값은 서버에서 측정한 모델 처리 시간입니다. 이를 이용하면 환경별로
가족 확인 모델, AI 음성 탐지 모델 중 어느 단계가 느린지 바로 확인할 수 있습니다.

### 통화 세션 시작

```bash
curl -X POST http://127.0.0.1:8000/api/v1/voice-sessions/start
```

### 통화 chunk 분석

```bash
curl -X POST http://127.0.0.1:8000/api/v1/voice-sessions/{session_id}/chunks \
  -F "audio_file=@samples/chunk_01.wav"
```

### 통화 세션 상태 확인

```bash
curl http://127.0.0.1:8000/api/v1/voice-sessions/{session_id}
```

## 실행 방법

### 1. Python 환경 준비

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Postgres 준비 (auth 등 신규 도메인용)

기존 SQLite 기반 기능(가족 음성 등록, 통화 세션 분석, 데모)은 그대로 동작하고, `auth` 도메인부터는 Postgres를 사용합니다.

```bash
docker compose up -d   # Postgres 컨테이너 기동
cp .env.example .env   # 최초 1회
alembic upgrade head   # 테이블 생성/마이그레이션 적용
```


### 3. FastAPI 서버 실행

```bash
uvicorn app.main:app --reload
```

서버 주소:

```text
http://127.0.0.1:8000
```

Swagger 문서:

```text
http://127.0.0.1:8000/docs
```


### 4. 모바일 데모 실행

```bash
cd mobile
npm install
npm run dev
```

모바일 데모 주소:

```text
http://127.0.0.1:5173
```

## 환경변수

설정은 `app/core/config.py`에 있으며 `ISFAM_` prefix로 덮어쓸 수 있습니다.

### Docker로 딥보이스 서버 실행

```bash
docker compose up --build ai postgres
```

기본 spectral 모델은 저장소에 포함되어 별도 모델 다운로드가 없습니다. Transformers
fallback을 선택한 첫 실행에서만 Hugging Face 모델을 내려받습니다. 기본 설정은 CPU
단일 worker와 프로세스 내부 최대 동시 추론 2개입니다.
실제 배포 인스턴스에서는 아래 부하 테스트를 다시 실행한 뒤 replica 수를 정합니다.

```bash
.venv/bin/python scripts/benchmark_deepvoice_api.py
```

```bash
ISFAM_SPEAKER_THRESHOLD=0.65 uvicorn app.main:app --reload
ISFAM_ANTI_SPOOFING_BACKEND=spectral uvicorn app.main:app --reload
ISFAM_ANTI_SPOOFING_THRESHOLD=0.5107068233191967 uvicorn app.main:app --reload
ISFAM_VOICE_SESSION_STRONG_SPOOF_SCORE=0.80 uvicorn app.main:app --reload
ISFAM_ANTI_SPOOFING_BATCH_SIZE=4 uvicorn app.main:app --reload
ISFAM_ANTI_SPOOFING_MAX_CONCURRENCY=2 uvicorn app.main:app --reload
ISFAM_ANTI_SPOOFING_MODEL_VERSION=2026-09-spectral-v1 uvicorn app.main:app --reload
ISFAM_PRELOAD_MODELS=true uvicorn app.main:app --reload
ISFAM_PRELOAD_SPEAKER_MODEL=false uvicorn app.main:app --reload
ISFAM_DEVICE=cpu uvicorn app.main:app --reload
```

주요 설정:

```text
ISFAM_SPEAKER_THRESHOLD
가족 voiceprint 유사도 기준

ISFAM_ANTI_SPOOFING_BACKEND
기본 spectral, 기존 Hugging Face 모델 fallback은 transformers

ISFAM_ANTI_SPOOFING_THRESHOLD
AI 합성 음성 탐지 기준

ISFAM_VOICE_SESSION_STRONG_SPOOF_SCORE
즉시 위험으로 볼 강한 spoof 기준

ISFAM_ANTI_SPOOFING_BATCH_SIZE
Transformers fallback에서 음성 구간을 한 번에 처리할 배치 크기, 기본값 4

ISFAM_ANTI_SPOOFING_MAX_CONCURRENCY
한 프로세스에서 동시에 실행할 딥보이스 추론 수, 기본값 2

ISFAM_ANTI_SPOOFING_MODEL_VERSION
운영 중인 딥보이스 모델의 배포 버전

ISFAM_PRELOAD_MODELS
서버 시작 시 두 AI 모델을 로드하고 워밍업할지 여부, 기본값 true

ISFAM_PRELOAD_SPEAKER_MODEL
딥보이스 전용 배포에서 가족 화자 모델 사전 로딩을 끌 수 있으며 기본값은 true

ISFAM_DATABASE_PATH
SQLite DB 경로, 기본값 data/isfam.sqlite3

ISFAM_DATABASE_URL
auth 등 Postgres 도메인용 접속 주소 (.env.example 참고)

ISFAM_JWT_SECRET_KEY
JWT 서명 키, 로컬 개발 기본값 그대로 두면 안 되고 배포 시 반드시 변경

ISFAM_DEVICE
cpu, cuda, auto 중 선택
```

## 프로젝트 구조

```text
app/
  api/routes/
    family.py              가족 등록 API
    voice.py               음성 비교 및 보안 검증 API
    voice_session.py       통화 chunk 누적 분석 API
    anti_spoofing.py       딥보이스 단독 탐지 API
  services/
    speaker_service.py     speaker embedding 추출 및 비교
    voiceprint_service.py  가족 voiceprint 등록/집계/검증
    anti_spoofing_service.py
    risk_scoring_service.py
  repositories/
    family_repository.py
    voice_session_repository.py
  db/
    session.py
mobile/
  src/
    main.js                모바일 통화 데모 로직
datasets/
  README.md
demo_samples/
  README.md
```

## 시연 시나리오

1. 가족 음성을 3개 이상 등록합니다.
2. 실제 가족 통화 음성을 업로드합니다.
3. 가족이 아닌 사람의 음성을 업로드합니다.
4. AI로 만든 가족 유사 음성을 업로드합니다.
5. 결과가 `safe`, `caution`, `danger`로 나뉘는지 확인합니다.
6. 응답의 `decision_reasons`로 판단 근거를 설명합니다.

평가 폴더의 등록 음성을 DB에 자동 등록:

```bash
.venv/bin/python scripts/register_eval_family.py
```

평가 폴더의 테스트 음성을 CSV로 분석:

```bash
.venv/bin/python scripts/evaluate_isfam_eval.py
```

현재 등록 가족 음성의 객관 지표 생성:

```bash
.venv/bin/python scripts/evaluate_family_voiceprints.py
```

생성되는 리포트:

```text
reports/family_voiceprint_summary.md
reports/family_voiceprint_threshold_metrics.csv
reports/family_voiceprint_pairs.csv
reports/family_voiceprint_sample_quality.csv
```

## 현재 한계와 개선 계획

현재 anti-spoofing 모델은 공개 DFADD 일부로 학습한 소형 모델입니다. 고정 벤치마크에서는 큰 개선을 확인했지만 화자 2명과 생성기 5종에 한정되므로 특정 한국어 통화 환경, 낮은 품질의 녹음, 재생 공격, 학습에 없던 생성 모델에서는 성능 편차가 생길 수 있습니다.

개선 계획:

```text
한국어·전화망·잡음·재생 공격 공개 평가셋 확장
미지 생성기 hold-out 평가
한국어 통화 데이터 기반 모델 비교
등록 문장 기반 active challenge 추가
위험도 점수 리포트 자동 생성
```

## 발표용 요약

IsFAM은 단일 AI 모델의 결과에 의존하지 않고, 가족 화자 인증 모델과 딥보이스 탐지 모델의 출력을 통화 구간별로 누적 분석하여 가족 사칭 위험도를 산출하는 하이브리드 AI 판별 시스템입니다. 다중 voiceprint 집계, median similarity 기반 안정화, AI 합성 점수의 보조 위험도 반영, 설명 가능한 decision reason을 통해 기존 오픈소스 활용을 넘어 서비스 도메인에 맞춘 독자적 최적화를 적용했습니다.
