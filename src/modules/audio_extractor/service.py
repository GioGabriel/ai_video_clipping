from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from src.core.command_runner import CommandRunner
from src.core.config import AppSettings

logger = logging.getLogger(__name__)


class AudioExtractor:
    def __init__(self, settings: AppSettings, command_runner: CommandRunner) -> None:
        self.settings = settings
        self.command_runner = command_runner

    def extract(
        self,
        video_path: Path,
        video_id: str,
        progress_callback: Callable[[str, int | None, int | None], None] | None = None,
    ) -> Path:
        audio_path = self.settings.audio_dir / f"{video_id}.wav"
        command = [
            self.settings.ffmpeg_binary,
            "-y",
            "-i",
            str(video_path),
            "-progress",
            "pipe:2",
            "-nostats",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ]
        if progress_callback is not None:
            progress_callback("FFmpeg audio extraction started.", None, None)
        self.command_runner.run(command, on_output=self._build_progress_handler(progress_callback))
        logger.info("Extracted audio for %s to %s", video_id, audio_path)
        return audio_path

    @staticmethod
    def _build_progress_handler(
        progress_callback: Callable[[str, int | None, int | None], None] | None,
    ) -> Callable[[str], None] | None:
        if progress_callback is None:
            return None

        emitted = {"started": False}

        def handle(line: str) -> None:
            message = line.strip()
            if not message:
                return
            if not emitted["started"] and (message.startswith("out_time=") or message.startswith("progress=")):
                emitted["started"] = True
                progress_callback("FFmpeg is writing the mono WAV audio track.", None, None)

        return handle
