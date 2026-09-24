# IsFAM AI 업그레이드 최종 결과

작성일: 2026-09-22  
범위: 전화환경 anti-spoofing, 음질 게이트, 애매 구간 정책, 속도 및 강건성 평가

## 한눈에 보는 결론

기본 딥보이스 탐지기를 `isfam/spectral-ensemble-telephone-v2`로 교체했다. 전화 대역과 음량 변화에 강한 v2가 자동 차단을 담당하고, 깨끗한 합성 음성에 강한 v1만 경고한 경우에는 위험으로 단정하지 않고 사용자 추가 확인으로 보낸다.

| 핵심 지표 | 기존 wav2vec2 | 최종 모델 | 변화 |
|---|---:|---:|---:|
| DFADD clean 정확도 | 50.88% | **99.38%** | +48.50%p |
| DFADD fake 자동 탐지 | 1.75% | **99.25%** | +97.50%p |
| DFADD real 자동 오탐 | 0.00% | **0.50%** | +0.50%p |
| DFADD fake 차단 또는 확인 | 1.75% | **100.00%** | +98.25%p |
| XTTS2 차단 또는 확인 | 1.15% | **97.70%** | +96.55%p |
| 평균 순수 추론 | 117.18 ms | **3.17 ms** | 37.0배 빠름 |
| p95 순수 추론 | 284.90 ms | **4.05 ms** | 70.4배 빠름 |
| 변환 포함 평균 | 149.68 ms | **31.45 ms** | 4.76배 빠름 |
| 모델 파일 | 378.30 MB | **40.88 KB** | 9,254배 작음 |

위 수치는 서버 CPU의 고정 평가셋 결과다. 실제 Android 전화망 정확도로 해석하면 안 된다.

## 최종 구조

```text
통화 음성
  ├─ 음질 검사
  │    길이 / 음량 / 발화 비율 / 추정 SNR 확인
  │    불량하면 모델 결과를 확정하지 않고 추가 음성 요청
  ├─ 온디바이스 가족 화자 인증
  │    ECAPA-TDNN INT8 ONNX → 등록 가족 목소리와 비교
  ├─ 서버 딥보이스 앙상블
  │    v2 전화 강건 모델 → 자동 차단 판단
  │    v1 clean 보조 모델 → v1만 경고하면 추가 확인
  └─ 위험도 및 통화 누적 판단
       safe / caution / danger
```

| 역할 | 현재 모델/정책 | 실행 위치 |
|---|---|---|
| 가족 화자 인증 | ECAPA-TDNN INT8 ONNX | Android 온디바이스 |
| 딥보이스 자동 판정 | `spectral-mlp-telephone-robust-v2` | FastAPI 서버 |
| 보조 경고 | `spectral-mlp-dfadd-v1` | FastAPI 서버 |
| 배포 artifact | `spectral-ensemble-telephone-v2` | FastAPI 서버 |
| 애매 구간 | threshold의 70~100%는 추가 확인 | 서버/통화 세션 |
| 저품질 음성 | 추정 SNR 25 dB 미만은 추가 음성 요청 | 서버 |

이번 교체는 서버 딥보이스 탐지 모델이다. 가족 화자 인증용 ONNX는 별도 모델이며 변경하지 않았다.

## 무엇을 바꿨나

### 전화에 강한 특징

- 입력의 DC 성분을 제거하고 RMS를 정규화해 통화 음량 차이의 영향을 줄였다.
- 전화에서 안정적인 400~3,200 Hz 음성 대역만 사용한다.
- 프레임별 전체 크기를 빼서 마이크 gain보다 스펙트럼 모양을 보게 했다.
- spectral flatness와 crest factor를 추가해 잡음성과 파형 찌그러짐을 구분한다.

### 전화환경 증강 재학습

원본 발화가 학습과 검증에 겹치지 않게 나눈 뒤 clean, 8 kHz 전화 대역, 여러 SNR의 백색/유색 잡음, -6~-24 dB 음량, clipping, 5~20% frame 손실, 잔향, 재생 모사를 적용했다. 학습 7,800개와 검증 2,600개 증강 예제를 사용했고, 검증 real 오탐 5% 이내에서 threshold `0.7839074731`을 선택했다.

### 보수적인 3단계 판단

```text
v2가 fake 판단           -> 자동 위험 판단
v2는 통과, v1만 경고     -> 추가 확인 필요
두 모델 모두 충분히 낮음 -> 정상 후보
음질 기준 미달           -> 모델 점수를 확정하지 않고 추가 음성 요청
```

통화 세션에서도 애매한 spoof 경고가 있으면 등록 가족으로 보이더라도 `low`로 내리지 않고 `medium / spoof_warning_needs_more_chunks`를 반환한다.

## 고정 clean 평가

DFADD test real 400개와 fake 400개를 사용했다. fake는 Grad-TTS, Matcha-TTS, NaturalSpeech2, PFlow-TTS, StyleTTS2 각 80개다.

| 운영 동작 | TP | TN | FP | FN | 정확도 | fake recall | real 오탐률 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 자동 fake, 기준 0.50 | 397 | 398 | 2 | 3 | **99.38%** | **99.25%** | **0.50%** |
| 차단 또는 확인, 기준 0.35 | 400 | 388 | 12 | 0 | 98.50% | **100.00%** | 3.00% 경고 |

두 번째 행의 3%는 모두 위험 확정 오탐이 아니라 추가 확인까지 포함한 경고율이다.

## 처음 보는 생성기

5개 생성기 중 하나를 학습에서 완전히 제외하고 나머지 4개로 학습한 뒤, 제외한 생성기를 시험했다.

| 제외한 생성기 | 자동 탐지 v2 | 차단 또는 확인 v1+v2 | real 경고율 v1+v2 |
|---|---:|---:|---:|
| Grad-TTS | 85.00% | 100.00% | 1.75% |
| Matcha-TTS | 91.25% | 95.00% | 2.25% |
| NaturalSpeech2 | 100.00% | 100.00% | 3.25% |
| PFlow-TTS | 100.00% | 100.00% | 2.50% |
| StyleTTS2 | 80.00% | 96.25% | 3.00% |
| **평균/최저** | **91.25% / 80.00%** | **98.25% / 95.00%** | **평균 2.55%** |

clean v1 단독의 이전 결과는 평균 69.25%, 최저 22.50%였다. 전화 강건 v2와 보조 확인 정책으로 미지 생성기 최저 포착률을 95%까지 올렸다.

## 보유 XTTS2 2,000개

| 동작 | 탐지/경고 수 | 비율 |
|---|---:|---:|
| 자동 fake 판정 | 1,438/2,000 | 71.90% |
| 자동 차단 또는 추가 확인 | 1,954/2,000 | **97.70%** |
| 놓침 | 46/2,000 | 2.30% |

이 세트는 fake-only이므로 정확도, precision, real 오탐률은 계산할 수 없다.

## 합성 전화환경 스트레스 시험

DFADD 800개 전체에 결정적 변형을 적용했다. `자동 분석률`은 추정 SNR 25 dB를 통과한 비율이고, 실패한 음성은 추가 확인으로 보낸다. 이는 실제 전화 녹음이 아닌 합성 변형 결과다.

| 조건 | raw fake 탐지 | raw real 오탐 | 자동 분석률 | 전체 추가 확인률 | fake 차단 또는 확인 |
|---|---:|---:|---:|---:|---:|
| clean | 99.25% | 0.50% | 97.50% | 4.13% | 100.00% |
| 8 kHz 전화 대역 | 98.75% | 0.25% | 100.00% | 0.50% | 99.25% |
| 잡음 20 dB | 85.00% | 3.75% | 13.25% | 92.38% | 100.00% |
| 잡음 10 dB | 67.25% | 8.25% | 0.00% | 100.00% | 100.00% |
| 작은 음량 -18 dB | 99.25% | 0.50% | 97.50% | 3.00% | 99.75% |
| 강한 clipping | 99.50% | 1.75% | 96.88% | 19.25% | 100.00% |
| 10% frame 손실 | 98.75% | 1.00% | 100.00% | 0.88% | 99.50% |
| 합성 잔향 | 95.75% | 1.75% | 92.00% | 23.13% | 99.75% |
| 재생 환경 모사 | 91.50% | 14.75% | 0.00% | 100.00% | 100.00% |

`raw` 열은 음질 게이트 전 모델 자체 수치다. 이전 v1은 8 kHz에서 fake 탐지 0%, 잡음에서 real 오탐 100%, 작은 음량에서 fake 탐지 15.25%였다. 새 v2는 8 kHz 98.75%, 작은 음량 99.25%로 회복했다. 잡음 10 dB와 재생 모사는 raw 모델 점수를 신뢰하지 않고 전부 사용자 확인으로 보내 오판 확정을 막는다.

강한 clipping과 잔향은 fake를 거의 놓치지 않지만 추가 확인률이 각각 19.25%, 23.13%라 사용자 불편이 남는다.

## 속도와 크기

최종 앙상블은 특징을 두 번 계산하므로 v1 단독 2.08 ms보다 느리지만, 기존 wav2vec2 117.18 ms보다 여전히 37배 빠르다.

| 지표 | 기존 wav2vec2 | 최종 앙상블 |
|---|---:|---:|
| 평균 추론 | 117.18 ms | 3.168 ms |
| p95 추론 | 284.90 ms | 4.045 ms |
| 변환 포함 평균 | 149.68 ms | 31.450 ms |
| 처리량 | 6.68 files/s | 31.69 files/s |
| artifact 크기 | 378,302,360 bytes | 40,880 bytes |

사용자 체감 시간에는 3~5초 음성 수집과 네트워크 업로드 시간이 별도로 포함된다.

## 배포 값

```text
ISFAM_ANTI_SPOOFING_BACKEND=spectral
ISFAM_ANTI_SPOOFING_MODEL_VERSION=2026-09-spectral-ensemble-v2
ISFAM_SPECTRAL_ANTI_SPOOFING_MODEL_PATH=app/assets/spectral_anti_spoof_ensemble_v2.npz
ISFAM_ANTI_SPOOFING_THRESHOLD=0.5
ISFAM_VOICE_SESSION_MIN_ESTIMATED_SNR_DB=25.0
```

최종 artifact SHA-256: `7ba7c94ce2cb1e57864750447b2e44e976a6b0f309be45b91e5ba8146eab80a4`

## 앱·앱 서버 연동

`IsFAM_BE`의 `dev` 브랜치와 `isfam-app`의 `main` 브랜치를 기준으로 실제 호출 경로까지 연결했다.

- Android 앱은 원본 m4a 전체 대신 전처리된 통화의 중앙 5초를 16 kHz mono PCM16 WAV로 만들어 전송한다. 전송량은 약 160 KB이며 앱 서버의 WAV 형식·1.5 MB 제한과 일치한다.
- 앱 서버는 `complete`, `additional_confirmation`, `more_voice_required`를 서로 다른 상태로 보존한다.
- 앱은 모델 점수에 자체 0.8 임계값을 다시 적용하지 않고 FastAPI가 반환한 `is_spoofed`와 `threshold`를 사용한다.
- `additional_confirmation`은 가족 화자와 일치해도 안전으로 내리지 않고 확인 필요로 표시한다.
- `more_voice_required`는 낮은 추정 SNR 등 음질 문제로 구분해 더 깨끗한 음성을 요청한다.
- FastAPI는 응답 직전 임시 업로드와 변환 WAV 삭제 성공 여부를 `purged=true/false`로 반환하고 앱 서버가 파기 기록에 보관한다.

검증은 FastAPI 테스트 22개, Android `testDebugUnitTest`·`assembleDebug`, Spring Boot 전체 테스트를 통과했다. 실제 FastAPI HTTP 호출에서도 `purged=true`, 모델명, 임계값, 추정 SNR 응답을 확인했다.

## 재현 방법

```bash
.venv/bin/pip install -r requirements-eval.txt
.venv/bin/python scripts/prepare_dfadd_eval.py
.venv/bin/python scripts/train_spectral_anti_spoofing.py
.venv/bin/python scripts/train_robust_spectral_anti_spoofing.py
.venv/bin/python scripts/build_spectral_ensemble.py
.venv/bin/python scripts/evaluate_anti_spoofing.py \
  --dataset datasets/public/dfadd/test \
  --thresholds 0.35,0.5
.venv/bin/python scripts/evaluate_spectral_robustness.py \
  --model-path app/assets/spectral_anti_spoof_ensemble_v2.npz
.venv/bin/python -m unittest discover -s tests -v
```

공개 음원과 원시 CSV/JSON 평가 결과는 용량과 라이선스 관리를 위해 git에서 제외한다.

## 남은 한계

- 실제 한국어 통화망, 휴대전화 마이크, 통신 codec, 스피커 재생·재녹음 데이터로 검증하지 못했다.
- DFADD는 화자 2명과 생성기 5종으로 작아 실제 서비스 분포를 대표하지 않는다.
- XTTS2의 자동 차단률은 71.90%이며, 나머지 대부분은 추가 확인 정책이 보완한다.
- 강한 clipping과 잔향에서는 추가 확인이 많아질 수 있다.
- 추정 SNR은 실제 계측기가 아니라 20 ms frame 에너지 백분위 기반의 보수적 품질 지표다.
- 긴 파일은 중앙 5초만 분석하므로 API 권장 입력은 연속된 3~5초 통화 chunk다.

따라서 현재 결과는 기존보다 훨씬 강한 개발 기준선이지만, 실제 출시 전에는 Android 실통화 A/B 검증이 마지막으로 필요하다.
