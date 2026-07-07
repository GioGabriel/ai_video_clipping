from pathlib import Path

import httpx
import pytest

from src.core.config import AppSettings
from src.core.models import SentenceBlock, TranscriptSegment
from src.modules.viral_detector.service import ViralMomentDetector


def build_settings(tmp_path: Path) -> AppSettings:
    data_dir = tmp_path / "data"
    return AppSettings(
        project_root=tmp_path,
        app_name="AI Clipping System",
        app_host="127.0.0.1",
        app_port=8000,
        log_level="INFO",
        data_dir=data_dir,
        videos_dir=data_dir / "videos",
        audio_dir=data_dir / "audio",
        transcripts_dir=data_dir / "transcripts",
        clips_dir=data_dir / "clips",
        database_path=data_dir / "app.db",
        yt_dlp_binary="yt-dlp",
        ffmpeg_binary="ffmpeg",
        yt_dlp_skip_certificate_check=False,
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="llama3",
        ollama_request_timeout_seconds=180,
        whisper_model="base",
        whisper_device="cpu",
        job_worker_count=1,
        max_clip_candidates=10,
        max_candidates_per_chunk=3,
        min_clip_duration_seconds=20,
        max_clip_duration_seconds=60,
        target_clip_duration_seconds=30,
        viral_chunk_duration_seconds=420,
        viral_chunk_overlap_seconds=60,
        viral_min_score=60,
        subtitle_generation_enabled=True,
    )


class StubResponse:
    def __init__(self, status_code: int, payload: dict[str, object]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, object]:
        return self._payload


class StubClient:
    def __init__(
        self,
        response: StubResponse | None = None,
        exception: Exception | None = None,
        expected_model: str = "llama3",
    ) -> None:
        self.response = response
        self.exception = exception
        self.expected_model = expected_model

    def post(self, path: str, json: dict[str, object]) -> StubResponse:
        if self.exception is not None:
            raise self.exception
        assert path == "/api/generate"
        assert json["model"] == self.expected_model
        assert json["stream"] is False
        assert self.response is not None
        return self.response


def test_detector_reports_missing_ollama_model(tmp_path: Path) -> None:
    detector = ViralMomentDetector(build_settings(tmp_path))
    detector._client = StubClient(  # type: ignore[assignment]
        response=StubResponse(404, {"error": "model 'llama3' not found"})
    )

    with pytest.raises(RuntimeError, match=r"ollama pull llama3"):
        detector.detect(
            [
                TranscriptSegment(
                    start_seconds=0,
                    end_seconds=30,
                    text="This is a test transcript segment with a strong opinion.",
                )
            ]
        )


def test_detector_reports_connection_failure(tmp_path: Path) -> None:
    detector = ViralMomentDetector(build_settings(tmp_path))
    detector._client = StubClient(  # type: ignore[assignment]
        exception=httpx.ConnectError("connection failed")
    )

    with pytest.raises(RuntimeError, match=r"Start it with `ollama serve`"):
        detector.detect(
            [
                TranscriptSegment(
                    start_seconds=0,
                    end_seconds=30,
                    text="This is a second test transcript segment with enough content.",
                )
            ]
        )


def test_detector_adds_heuristic_candidate_for_quotable_opening(tmp_path: Path) -> None:
    detector = ViralMomentDetector(build_settings(tmp_path))
    detector._client = StubClient(  # type: ignore[assignment]
        response=StubResponse(200, {"response": '{"moments": []}'})
    )

    moments = detector.detect(
        [
            TranscriptSegment(
                start_seconds=0.0,
                end_seconds=4.5,
                text="Everyone is jealous of what you've got. No one is jealous of how you got it.",
            ),
            TranscriptSegment(
                start_seconds=4.5,
                end_seconds=6.4,
                text="I love that quote. My God.",
            ),
            TranscriptSegment(
                start_seconds=6.4,
                end_seconds=11.2,
                text="People see the trophies, but not the training ground. Everybody wants the view, but no one wants the climb.",
            ),
        ]
    )

    assert moments
    assert any(moment.start_seconds <= 0.0 and "jealous" in moment.hook.lower() for moment in moments)


def test_detector_prefers_earlier_motivational_thesis_over_late_summary(tmp_path: Path) -> None:
    detector = ViralMomentDetector(build_settings(tmp_path))
    detector._client = StubClient(  # type: ignore[assignment]
        response=StubResponse(200, {"response": '{"moments": []}'})
    )

    moments = detector.detect(
        [
            TranscriptSegment(
                start_seconds=0.0,
                end_seconds=4.0,
                text="Why does discipline equal freedom?",
            ),
            TranscriptSegment(
                start_seconds=4.0,
                end_seconds=9.0,
                text="Because the more discipline you have in your life, the more freedom you will end up with.",
            ),
            TranscriptSegment(
                start_seconds=9.0,
                end_seconds=14.0,
                text="If you lack the discipline to manage your time, you will end up with no free time.",
            ),
            TranscriptSegment(
                start_seconds=14.0,
                end_seconds=18.0,
                text="And if you do that, you're going to end up with freedom across the board.",
            ),
        ]
    )

    assert moments
    assert any(moment.start_seconds <= 0.0 and "discipline" in moment.hook.lower() for moment in moments)
    assert all(moment.duration_seconds >= 20.0 for moment in moments)


def test_detector_finds_motivational_thesis_candidates_without_llm_hits(tmp_path: Path) -> None:
    detector = ViralMomentDetector(build_settings(tmp_path))
    detector._client = StubClient(  # type: ignore[assignment]
        response=StubResponse(200, {"response": '{"moments": []}'})
    )

    moments = detector.detect(
        [
            TranscriptSegment(
                start_seconds=0.0,
                end_seconds=6.0,
                text="Giving up is the easiest thing to do in the world.",
            ),
            TranscriptSegment(
                start_seconds=6.0,
                end_seconds=13.0,
                text="Your life is either hard now and easy later or easy now and hard later.",
            ),
            TranscriptSegment(
                start_seconds=13.0,
                end_seconds=19.0,
                text="Train yourself for what is hard.",
            ),
            TranscriptSegment(
                start_seconds=19.0,
                end_seconds=27.0,
                text="Do what needs to be done regardless of how you feel.",
            ),
            TranscriptSegment(
                start_seconds=27.0,
                end_seconds=34.0,
                text="When your why is strong enough, your how will reveal itself.",
            ),
        ]
    )

    assert moments
    assert any("hard now" in moment.hook.lower() or "giving up" in moment.hook.lower() for moment in moments)
    assert any(moment.phrase_score > 0 for moment in moments)
    assert all(moment.score >= detector.settings.viral_min_score for moment in moments)


def test_detector_carries_phrase_score_for_strong_hook_text(tmp_path: Path) -> None:
    detector = ViralMomentDetector(build_settings(tmp_path))
    detector._client = StubClient(  # type: ignore[assignment]
        response=StubResponse(
            200,
            {
                "response": (
                    '{"moments":[{"hook":"Most people do this wrong.","start_time":"00:00:00",'
                    '"end_time":"00:00:25","reason":"Strong hook"}]}'
                )
            },
        )
    )

    moments = detector.detect(
        [
            TranscriptSegment(
                start_seconds=0.0,
                end_seconds=25.0,
                text="Most people do this wrong. Here is what to do next.",
            )
        ]
    )

    assert moments
    assert moments[0].phrase_score > 0
    assert moments[0].score >= detector.settings.viral_min_score
    assert "most people" in moments[0].hook.lower()


def test_detector_uses_job_selected_model_for_ollama_request(tmp_path: Path) -> None:
    detector = ViralMomentDetector(build_settings(tmp_path))
    detector._client = StubClient(  # type: ignore[assignment]
        response=StubResponse(200, {"response": '{"moments": []}'}),
        expected_model="qwen2.5:7b",
    )

    detector.detect(
        [
            SentenceBlock(
                start_seconds=0.0,
                end_seconds=25.0,
                text="Most people do this wrong. Here is what to do next.",
                speaker="speaker_1",
            )
        ],
        model_name="qwen2.5:7b",
    )
