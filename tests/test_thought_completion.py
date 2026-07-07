from pathlib import Path

from src.core.config import AppSettings
from src.core.models import SentenceBlock, ViralMoment
from src.modules.thought_completion.service import ThoughtCompletion


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


def test_thought_completion_extends_to_finish_related_sentence(tmp_path: Path) -> None:
    completer = ThoughtCompletion(build_settings(tmp_path))
    blocks = [
        SentenceBlock(0.0, 6.0, "If you lack the discipline to exercise and eat healthy,"),
        SentenceBlock(6.0, 12.0, "you will end up being a slave to disease."),
        SentenceBlock(13.5, 18.0, "Now we are changing topics."),
    ]
    moments = [
        ViralMoment(
            start_seconds=0.0,
            end_seconds=6.0,
            score=78,
            hook="If you lack discipline",
            reason="Cause and effect advice",
            core_start_seconds=0.0,
            core_end_seconds=6.0,
        )
    ]

    completed = completer.complete(blocks, moments)

    assert completed[0].end_seconds >= 12.0
    assert completed[0].end_seconds < 18.5
