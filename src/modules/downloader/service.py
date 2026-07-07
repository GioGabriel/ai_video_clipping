from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

from src.core.command_runner import CommandError, CommandRunner
from src.core.config import AppSettings

logger = logging.getLogger(__name__)


class VideoDownloader:
    def __init__(self, settings: AppSettings, command_runner: CommandRunner) -> None:
        self.settings = settings
        self.command_runner = command_runner

    def download(
        self,
        url: str,
        video_id: str,
        progress_callback: Callable[[str, int | None, int | None], None] | None = None,
    ) -> Path:
        output_template = self.settings.videos_dir / f"{video_id}.%(ext)s"
        command = [
            self.settings.yt_dlp_binary,
            "--no-playlist",
            "--newline",
            "-f",
            self.settings.download_format,
            "--merge-output-format",
            "mp4",
            "-o",
            str(output_template),
            url,
        ]
        if self.settings.yt_dlp_skip_certificate_check:
            command.insert(1, "--no-check-certificates")

        try:
            self.command_runner.run(command, on_output=self._build_progress_handler(progress_callback))
        except CommandError as exc:
            error_message = str(exc)
            if "CERTIFICATE_VERIFY_FAILED" in error_message:
                raise RuntimeError(
                    "yt-dlp failed TLS certificate verification while downloading the source video. "
                    "If you are on a trusted network with SSL interception, set "
                    "`YT_DLP_SKIP_CERTIFICATE_CHECK=true` in `.env` and retry. "
                    "Otherwise fix the local certificate trust chain."
                ) from exc
            raise

        candidates = sorted(
            path
            for path in self.settings.videos_dir.glob(f"{video_id}.*")
            if path.suffix not in {".part", ".ytdl"}
        )
        if not candidates:
            raise RuntimeError("yt-dlp did not produce an output video file.")

        preferred = next((path for path in candidates if path.suffix == ".mp4"), candidates[0])
        logger.info("Downloaded %s to %s", url, preferred)
        return preferred

    @staticmethod
    def _build_progress_handler(
        progress_callback: Callable[[str, int | None, int | None], None] | None,
    ) -> Callable[[str], None] | None:
        if progress_callback is None:
            return None

        last_bucket = {"value": -10}

        def handle(line: str) -> None:
            message = line.strip()
            if not message:
                return

            lowered = message.lower()
            if "destination:" in lowered:
                progress_callback(message, None, None)
                return
            if message.startswith("[Merger]") or message.startswith("[ExtractAudio]"):
                progress_callback(message, None, None)
                return

            match = re.search(r"(\d+(?:\.\d+)?)%", message)
            if match:
                bucket = int(float(match.group(1)) // 5) * 5
                if bucket > last_bucket["value"]:
                    last_bucket["value"] = bucket
                    progress_callback(f"yt-dlp download progress: {bucket}%.", None, None)

        return handle
