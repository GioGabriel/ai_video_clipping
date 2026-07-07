from pathlib import Path

from src.core.config import AppSettings
from src.core.models import SentenceBlock, ViralMoment
from src.modules.clip_optimizer.service import ClipOptimizer


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


def test_clip_optimizer_expands_short_clip_to_meet_minimum(tmp_path: Path) -> None:
    optimizer = ClipOptimizer(build_settings(tmp_path))
    blocks = [
        SentenceBlock(0.0, 6.0, "Most people are doing this wrong."),
        SentenceBlock(6.0, 13.0, "This mistake costs companies millions."),
        SentenceBlock(13.0, 21.0, "Here is how you fix it."),
    ]
    moments = [
        ViralMoment(
            start_seconds=0.0,
            end_seconds=6.0,
            score=84,
            hook="Most people are doing this wrong",
            reason="Strong hook",
            core_start_seconds=0.0,
            core_end_seconds=6.0,
            duration_score=0.4,
        )
    ]

    optimized = optimizer.optimize(blocks, moments)

    assert len(optimized) == 1
    assert optimized[0].duration_seconds >= 18.0


def test_clip_optimizer_splits_overlong_clip_into_short_form_chunks(tmp_path: Path) -> None:
    optimizer = ClipOptimizer(build_settings(tmp_path))
    blocks = [
        SentenceBlock(index * 10.0, (index * 10.0) + 10.0, f"Sentence block {index}.")
        for index in range(7)
    ]
    moments = [
        ViralMoment(
            start_seconds=0.0,
            end_seconds=70.0,
            score=88,
            hook="Sentence block 0",
            reason="Long explainer",
            core_start_seconds=0.0,
            core_end_seconds=70.0,
            duration_score=0.2,
        )
    ]

    optimized = optimizer.optimize(blocks, moments)

    assert len(optimized) >= 2
    assert all(moment.duration_seconds <= 50.0 for moment in optimized)
