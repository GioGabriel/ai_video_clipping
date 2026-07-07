from pathlib import Path

from src.core.config import AppSettings
from src.core.models import SentenceBlock, ViralMoment
from src.modules.hook_detector.service import HookDetector


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
        min_clip_duration_seconds=18,
        max_clip_duration_seconds=50,
        target_clip_duration_seconds=30,
        viral_chunk_duration_seconds=420,
        viral_chunk_overlap_seconds=60,
        viral_min_score=60,
        subtitle_generation_enabled=True,
    )


def test_hook_detector_rewinds_to_stronger_opening_sentence(tmp_path: Path) -> None:
    detector = HookDetector(build_settings(tmp_path))
    blocks = [
        SentenceBlock(0.0, 4.0, "Why does discipline equal freedom?"),
        SentenceBlock(4.0, 10.0, "Because the more discipline you have in your life, the more freedom you end up with."),
        SentenceBlock(10.0, 14.0, "And if you do that, you're going to end up with freedom across the board."),
    ]
    moments = [
        ViralMoment(
            start_seconds=10.0,
            end_seconds=14.0,
            score=82,
            hook="freedom across the board",
            reason="Strong payoff",
            core_start_seconds=10.0,
            core_end_seconds=14.0,
        )
    ]

    detected = detector.detect(blocks, moments)

    assert detected[0].start_seconds == 0.0
    assert "discipline equal freedom" in detected[0].hook.lower()


def test_hook_detector_keeps_original_start_when_earlier_line_is_weaker(tmp_path: Path) -> None:
    detector = HookDetector(build_settings(tmp_path))
    blocks = [
        SentenceBlock(1.0, 3.0, "Welcome back everyone."),
        SentenceBlock(4.0, 7.0, "Why most people fail at this?", phrase_score=20.0, detected_triggers=["curiosity"]),
    ]
    moments = [
        ViralMoment(
            start_seconds=4.0,
            end_seconds=10.0,
            score=84,
            hook="Why most people fail at this?",
            reason="Strong question hook",
            hook_start_seconds=4.0,
            core_start_seconds=4.0,
            core_end_seconds=10.0,
        )
    ]

    detected = detector.detect(blocks, moments)

    assert detected[0].start_seconds == 4.0
    assert detected[0].hook_start_seconds == 4.0
    assert detected[0].hook == "Why most people fail at this?"
