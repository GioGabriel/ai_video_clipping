from pathlib import Path

from src.core.config import AppSettings
from src.core.database import Database
from src.core.repositories import JobRepository


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


def test_reconcile_incomplete_jobs_marks_queued_and_running_as_failed(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings.ensure_directories()

    database = Database(settings.database_path)
    database.initialize()
    job_repository = JobRepository(database)

    job_repository.create(job_id="queued-job", video_id="video-queued", source_url="https://example.com/q")
    job_repository.create(job_id="running-job", video_id="video-running", source_url="https://example.com/r")
    job_repository.update("running-job", status="running", current_step="download_video")

    reconciled = job_repository.reconcile_incomplete_jobs("worker interrupted")

    assert {job["id"] for job in reconciled} == {"queued-job", "running-job"}
    assert job_repository.get("queued-job")["status"] == "failed"
    assert job_repository.get("running-job")["status"] == "failed"
    assert job_repository.get("running-job")["current_step"] == "download_video"
    assert job_repository.get("queued-job")["error_message"] == "worker interrupted"
