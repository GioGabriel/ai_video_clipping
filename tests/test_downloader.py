from pathlib import Path

import pytest

from src.core.command_runner import CommandError
from src.core.config import AppSettings
from src.modules.downloader.service import VideoDownloader


class CapturingRunner:
    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.commands: list[list[str]] = []

    def run(self, command: list[str], cwd: Path | None = None, on_output=None) -> None:  # noqa: ANN001
        self.commands.append(list(command))
        if on_output is not None:
            on_output("[download]  50.0% of 10.00MiB")
        if self.should_fail:
            raise CommandError(
                "Command failed (1): yt-dlp\n"
                "<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed>"
            )


def build_settings(tmp_path: Path, *, skip_cert_check: bool = False) -> AppSettings:
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
        yt_dlp_skip_certificate_check=skip_cert_check,
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


def test_downloader_includes_no_check_certificates_when_enabled(tmp_path: Path) -> None:
    settings = build_settings(tmp_path, skip_cert_check=True)
    settings.ensure_directories()
    output_file = settings.videos_dir / "video123.mp4"
    output_file.write_text("placeholder", encoding="utf-8")

    runner = CapturingRunner()
    downloader = VideoDownloader(settings, runner)

    result = downloader.download("https://example.com/video", "video123")

    assert result == output_file
    assert "--no-check-certificates" in runner.commands[0]


def test_downloader_raises_clear_message_for_tls_errors(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings.ensure_directories()
    runner = CapturingRunner(should_fail=True)
    downloader = VideoDownloader(settings, runner)

    with pytest.raises(RuntimeError, match="YT_DLP_SKIP_CERTIFICATE_CHECK=true"):
        downloader.download("https://example.com/video", "video123")
