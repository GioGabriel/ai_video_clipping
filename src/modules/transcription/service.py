from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path

from src.core.config import AppSettings
from src.core.models import TranscriptArtifact, TranscriptSegment, TranscriptWord
from src.core.timecode import seconds_to_timecode

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._model = None
        self._model_lock = threading.Lock()

    def transcribe(
        self,
        audio_path: Path,
        video_id: str,
        progress_callback: Callable[[str, int | None, int | None], None] | None = None,
    ) -> TranscriptArtifact:
        if progress_callback is not None:
            progress_callback(
                f"Loading Whisper model `{self.settings.whisper_model}` on `{self.settings.whisper_device}`.",
                None,
                None,
            )
        model = self._get_model()
        if progress_callback is not None:
            progress_callback("Whisper model loaded. Running transcription with word timestamps enabled.", None, None)
        use_fp16 = self.settings.whisper_device.lower() != "cpu"
        result = model.transcribe(str(audio_path), verbose=False, fp16=use_fp16, word_timestamps=True)

        segments: list[TranscriptSegment] = []
        for segment in result.get("segments", []):
            if not segment.get("text", "").strip():
                continue

            words = [
                TranscriptWord(
                    start_seconds=float(word.get("start", segment["start"])),
                    end_seconds=float(word.get("end", segment["end"])),
                    text=str(word.get("word", "")).strip(),
                )
                for word in segment.get("words", [])
                if str(word.get("word", "")).strip()
            ]
            segments.append(
                TranscriptSegment(
                    start_seconds=float(segment["start"]),
                    end_seconds=float(segment["end"]),
                    text=segment["text"].strip(),
                    words=words or None,
                )
            )

        transcript_path = self.settings.transcripts_dir / f"{video_id}.json"
        payload = {
            "video_id": video_id,
            "language": result.get("language", "unknown"),
            "text": result.get("text", "").strip(),
            "segments": [
                {
                    "start": seconds_to_timecode(segment.start_seconds),
                    "end": seconds_to_timecode(segment.end_seconds),
                    "start_seconds": segment.start_seconds,
                    "end_seconds": segment.end_seconds,
                    "text": segment.text,
                    "words": [
                        {
                            "start": seconds_to_timecode(word.start_seconds),
                            "end": seconds_to_timecode(word.end_seconds),
                            "start_seconds": word.start_seconds,
                            "end_seconds": word.end_seconds,
                            "text": word.text,
                        }
                        for word in (segment.words or [])
                    ],
                }
                for segment in segments
            ],
        }
        transcript_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved transcript for %s to %s", video_id, transcript_path)
        if progress_callback is not None:
            progress_callback(f"Whisper produced {len(segments)} transcript segment(s).", len(segments), len(segments))

        return TranscriptArtifact(
            video_id=video_id,
            language=payload["language"],
            text=payload["text"],
            transcript_path=transcript_path,
            segments=segments,
        )

    def _get_model(self):
        with self._model_lock:
            if self._model is not None:
                return self._model

            try:
                import whisper
            except ImportError as exc:
                raise RuntimeError(
                    "The `openai-whisper` package is required for transcription."
                ) from exc

            logger.info("Loading Whisper model `%s` on `%s`.", self.settings.whisper_model, self.settings.whisper_device)
            self._model = whisper.load_model(self.settings.whisper_model, device=self.settings.whisper_device)
            return self._model
