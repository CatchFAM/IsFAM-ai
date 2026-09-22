import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, HTMLResponse

from app.core.config import Settings, get_settings
from app.repositories.demo_session_repository import DemoSessionRepository
from app.schemas.anti_spoofing import AntiSpoofingLabelScore, AntiSpoofingResponse
from app.schemas.demo import DemoAnswerRequest, DemoAnswerResponse, DemoStartResponse
from app.services.anti_spoofing_service import AntiSpoofingError, AntiSpoofingResult
from app.services.demo_service import (
    DemoSampleFileMissingError,
    DemoSampleNotFoundError,
    DemoService,
    DemoSessionNotFoundError,
)
from app.services.model_provider import get_anti_spoofing_service
from app.utils.audio import (
    AudioDecodingError,
    AudioTooShortError,
    AudioValidationError,
    cleanup_temp_files,
    convert_audio_to_standard_wav,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/demo-sessions", tags=["demo"])
web_router = APIRouter(tags=["demo"])


def get_demo_repository(settings: Settings = Depends(get_settings)) -> DemoSessionRepository:
    return DemoSessionRepository(settings.database_path)


def get_demo_service(
    settings: Settings = Depends(get_settings),
    demo_repository: DemoSessionRepository = Depends(get_demo_repository),
) -> DemoService:
    return DemoService(
        settings=settings,
        demo_session_repository=demo_repository,
    )


def _anti_spoofing_result_to_response(result: AntiSpoofingResult) -> AntiSpoofingResponse:
    return AntiSpoofingResponse(
        analysis_status=(
            "additional_confirmation"
            if result.message == "additional_confirmation"
            else "complete"
        ),
        is_spoofed=result.is_spoofed,
        spoof_score=result.spoof_score,
        threshold=result.threshold,
        predicted_label=result.predicted_label,
        predicted_score=result.predicted_score,
        message=result.message,
        model_name=result.model_name,
        analyzed_segments=result.analyzed_segments,
        max_spoof_segment_index=result.max_spoof_segment_index,
        segment_seconds=result.segment_seconds,
        label_scores=[
            AntiSpoofingLabelScore(label=label_score.label, score=label_score.score)
            for label_score in result.label_scores
        ],
    )


def _media_type_for_audio(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix == ".m4a":
        return "audio/mp4"
    return "application/octet-stream"


@web_router.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "IsFAM API server is running. Use the mobile demo at http://127.0.0.1:5173.",
    }


@web_router.get("/demo", response_class=HTMLResponse, include_in_schema=False)
async def demo_page() -> HTMLResponse:
    return HTMLResponse(DEMO_HTML)


@router.post("/start", response_model=DemoStartResponse)
async def start_demo_session(
    service: DemoService = Depends(get_demo_service),
) -> DemoStartResponse:
    """Start a no-cost demo session by choosing a local real/fake sample."""

    try:
        result = service.start_demo_session()
        return DemoStartResponse(
            session_id=result.session_id,
            audio_url=result.audio_url,
            playback_seconds=result.playback_seconds,
            message=result.message,
        )
    except DemoSampleNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while starting demo session")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="internal server error while starting demo session",
        ) from exc


@router.get("/{session_id}/audio")
async def get_demo_audio(
    session_id: str,
    service: DemoService = Depends(get_demo_service),
) -> FileResponse:
    """Stream the selected sample without exposing whether it is real or fake."""

    try:
        record = service.get_session(session_id)
        return FileResponse(
            path=record.sample_path,
            media_type=_media_type_for_audio(record.sample_path),
            filename=f"isfam-demo{record.sample_path.suffix.lower()}",
        )
    except DemoSessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except DemoSampleFileMissingError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while reading demo audio")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="internal server error while reading demo audio",
        ) from exc


@router.post("/{session_id}/answer", response_model=DemoAnswerResponse)
async def answer_demo_session(
    session_id: str,
    request: DemoAnswerRequest,
    settings: Settings = Depends(get_settings),
    service: DemoService = Depends(get_demo_service),
) -> DemoAnswerResponse:
    """Evaluate the user guess and compare it with IsFAM AI judgment."""

    temp_paths: list[Path | None] = []

    try:
        record = service.submit_answer(
            session_id=session_id,
            user_guess=request.user_guess,
        )

        wav_file = convert_audio_to_standard_wav(
            input_path=record.sample_path,
            target_sample_rate=settings.target_sample_rate,
            min_audio_seconds=settings.min_audio_seconds,
        )
        temp_paths.append(wav_file)

        anti_spoofing_result = get_anti_spoofing_service().detect_file(wav_file)
        ai_guess = anti_spoofing_result.predicted_label

        return DemoAnswerResponse(
            session_id=record.id,
            user_guess=record.user_guess or request.user_guess,
            actual_label=record.actual_label,
            is_user_correct=(record.user_guess == record.actual_label),
            ai_guess=ai_guess,
            is_ai_correct=(ai_guess == record.actual_label),
            anti_spoofing=_anti_spoofing_result_to_response(anti_spoofing_result),
            message="demo_answer_evaluated",
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except DemoSessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except DemoSampleFileMissingError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except AudioTooShortError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except AudioDecodingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except AudioValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except AntiSpoofingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error while answering demo session")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="internal server error while answering demo session",
        ) from exc
    finally:
        cleanup_temp_files(temp_paths)


DEMO_HTML = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>IsFAM Demo</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --text: #171a21;
      --muted: #667085;
      --line: #d9dee8;
      --brand: #0f766e;
      --brand-dark: #115e59;
      --warn: #b45309;
      --danger: #b42318;
      --ok: #047857;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(920px, calc(100vw - 32px));
      margin: 32px auto;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 20px;
    }
    h1 {
      margin: 0;
      font-size: 28px;
      letter-spacing: 0;
    }
    .sub {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 16px;
    }
    .row {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }
    button {
      border: 0;
      border-radius: 6px;
      background: var(--brand);
      color: white;
      padding: 11px 14px;
      font-weight: 700;
      cursor: pointer;
      min-height: 42px;
    }
    button:hover { background: var(--brand-dark); }
    button.secondary {
      background: #e7eef6;
      color: #1f2937;
    }
    button.secondary:hover { background: #dbe5f0; }
    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }
    audio {
      width: 100%;
      margin-top: 16px;
    }
    .status {
      min-height: 24px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
    }
    .hidden { display: none; }
    .choices {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }
    .choices button {
      min-height: 64px;
      font-size: 16px;
    }
    .result-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fbfcfe;
      min-height: 92px;
    }
    .metric strong {
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 8px;
    }
    .metric span {
      display: block;
      font-size: 20px;
      font-weight: 800;
      word-break: keep-all;
    }
    .ok { color: var(--ok); }
    .bad { color: var(--danger); }
    .warn { color: var(--warn); }
    pre {
      overflow: auto;
      background: #111827;
      color: #e5e7eb;
      border-radius: 8px;
      padding: 14px;
      font-size: 13px;
      line-height: 1.5;
    }
    @media (max-width: 720px) {
      header { align-items: flex-start; flex-direction: column; }
      .choices, .result-grid { grid-template-columns: 1fr; }
      h1 { font-size: 24px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>IsFAM 음성 판별 데모</h1>
        <p class="sub">로컬 샘플을 랜덤 재생하고 사용자의 판단과 IsFAM AI 판단을 비교합니다.</p>
      </div>
      <a href="/docs">API Docs</a>
    </header>

    <section class="panel">
      <div class="row">
        <button id="startBtn" type="button">랜덤 음성 시작</button>
        <button id="resetBtn" class="secondary" type="button">초기화</button>
      </div>
      <p id="status" class="status">시작 버튼을 누르면 실제 녹음 또는 AI 샘플 중 하나가 재생됩니다.</p>
      <audio id="audio" controls class="hidden"></audio>
    </section>

    <section id="question" class="panel hidden">
      <h2>이 음성은 무엇일까요?</h2>
      <div class="choices">
        <button id="realBtn" type="button">실제 음성</button>
        <button id="fakeBtn" type="button">AI 음성</button>
      </div>
    </section>

    <section id="result" class="panel hidden">
      <h2>결과</h2>
      <div class="result-grid">
        <div class="metric">
          <strong>사용자 판단</strong>
          <span id="userGuess"></span>
        </div>
        <div class="metric">
          <strong>정답</strong>
          <span id="actualLabel"></span>
        </div>
        <div class="metric">
          <strong>IsFAM AI 판단</strong>
          <span id="aiGuess"></span>
        </div>
      </div>
      <div class="result-grid">
        <div class="metric">
          <strong>spoof_score</strong>
          <span id="spoofScore"></span>
        </div>
        <div class="metric">
          <strong>threshold</strong>
          <span id="threshold"></span>
        </div>
        <div class="metric">
          <strong>model</strong>
          <span id="modelName"></span>
        </div>
      </div>
      <pre id="rawJson"></pre>
    </section>
  </main>

  <script>
    let currentSessionId = null;
    let promptTimer = null;

    const startBtn = document.getElementById("startBtn");
    const resetBtn = document.getElementById("resetBtn");
    const statusEl = document.getElementById("status");
    const audioEl = document.getElementById("audio");
    const questionEl = document.getElementById("question");
    const resultEl = document.getElementById("result");

    function labelText(value) {
      if (value === "fake") return "AI 음성";
      if (value === "uncertain") return "추가 확인 필요";
      return "실제 음성";
    }

    function setStatus(message) {
      statusEl.textContent = message;
    }

    function setBusy(isBusy) {
      startBtn.disabled = isBusy;
      document.getElementById("realBtn").disabled = isBusy;
      document.getElementById("fakeBtn").disabled = isBusy;
    }

    function reset() {
      currentSessionId = null;
      if (promptTimer) clearTimeout(promptTimer);
      audioEl.pause();
      audioEl.removeAttribute("src");
      audioEl.classList.add("hidden");
      questionEl.classList.add("hidden");
      resultEl.classList.add("hidden");
      setStatus("시작 버튼을 누르면 실제 녹음 또는 AI 샘플 중 하나가 재생됩니다.");
      setBusy(false);
    }

    async function startDemo() {
      reset();
      setBusy(true);
      setStatus("샘플을 준비하는 중입니다.");

      try {
        const response = await fetch("/api/v1/demo-sessions/start", { method: "POST" });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "데모 세션을 시작하지 못했습니다.");
        }

        currentSessionId = data.session_id;
        audioEl.src = data.audio_url;
        audioEl.classList.remove("hidden");
        await audioEl.play().catch(() => null);
        setStatus("음성을 듣고 잠시 뒤 질문에 답해주세요.");

        promptTimer = setTimeout(() => {
          questionEl.classList.remove("hidden");
          setStatus("실제 음성인지 AI 음성인지 선택하세요.");
        }, data.playback_seconds * 1000);
      } catch (error) {
        setStatus(error.message);
      } finally {
        setBusy(false);
      }
    }

    async function answerDemo(userGuess) {
      if (!currentSessionId) return;
      setBusy(true);
      setStatus("IsFAM AI가 같은 샘플을 분석하는 중입니다.");

      try {
        const response = await fetch(`/api/v1/demo-sessions/${currentSessionId}/answer`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_guess: userGuess })
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "답변을 평가하지 못했습니다.");
        }

        document.getElementById("userGuess").textContent = labelText(data.user_guess);
        document.getElementById("actualLabel").textContent = labelText(data.actual_label);
        document.getElementById("aiGuess").textContent = labelText(data.ai_guess);
        document.getElementById("spoofScore").textContent = data.anti_spoofing.spoof_score;
        document.getElementById("threshold").textContent = data.anti_spoofing.threshold;
        document.getElementById("modelName").textContent = data.anti_spoofing.model_name;
        document.getElementById("rawJson").textContent = JSON.stringify(data, null, 2);

        document.getElementById("userGuess").className = data.is_user_correct ? "ok" : "bad";
        document.getElementById("aiGuess").className = data.is_ai_correct ? "ok" : "bad";
        resultEl.classList.remove("hidden");
        setStatus("결과가 나왔습니다.");
      } catch (error) {
        setStatus(error.message);
      } finally {
        setBusy(false);
      }
    }

    startBtn.addEventListener("click", startDemo);
    resetBtn.addEventListener("click", reset);
    document.getElementById("realBtn").addEventListener("click", () => answerDemo("real"));
    document.getElementById("fakeBtn").addEventListener("click", () => answerDemo("fake"));
  </script>
</body>
</html>
"""
