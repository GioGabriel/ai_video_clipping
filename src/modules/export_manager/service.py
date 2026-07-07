from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.core.config import AppSettings
from src.core.models import CaptionTheme, ClipArtifact, EditPlan, OutputAspectRatio, TranscriptArtifact, ViralMoment
from src.core.timecode import seconds_to_timecode


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExportManager:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def build_manifest(
        self,
        job_id: str,
        video_id: str,
        source_url: str,
        output_aspect_ratio: OutputAspectRatio | str,
        caption_theme: CaptionTheme | str,
        transcript: TranscriptArtifact,
        moments: list[ViralMoment],
        clips: list[ClipArtifact],
        edit_plans: list[EditPlan],
    ) -> Path:
        clip_directory = self.settings.clips_dir / video_id
        clip_directory.mkdir(parents=True, exist_ok=True)
        manifest_path = clip_directory / "manifest.json"

        payload = {
            "job_id": job_id,
            "video_id": video_id,
            "source_url": source_url,
            "output_aspect_ratio": str(output_aspect_ratio),
            "caption_theme": str(caption_theme),
            "generated_at": _utcnow_iso(),
            "transcript_path": str(transcript.transcript_path),
            "candidate_moments": [
                {
                    "start": seconds_to_timecode(moment.start_seconds),
                    "end": seconds_to_timecode(moment.end_seconds),
                    "start_seconds": moment.start_seconds,
                    "end_seconds": moment.end_seconds,
                    "duration_seconds": moment.duration_seconds,
                    "score": moment.score,
                    "hook": moment.hook,
                    "reason": moment.reason,
                    "hook_start_seconds": moment.hook_start_seconds,
                    "core_start_seconds": moment.core_start_seconds,
                    "core_end_seconds": moment.core_end_seconds,
                    "hook_strength": moment.hook_strength,
                    "emotion_level": moment.emotion_level,
                    "statement_strength": moment.statement_strength,
                    "novelty": moment.novelty,
                    "duration_score": moment.duration_score,
                    "phrase_score": moment.phrase_score,
                }
                for moment in moments
            ],
            "clips": [
                {
                    "clip_id": f"{video_id}_clip_{clip.sequence_number:03d}",
                    "sequence_number": clip.sequence_number,
                    "file_path": str(clip.file_path),
                    "subtitle_path": str(clip.subtitle_path) if clip.subtitle_path else None,
                    "styled_subtitle_path": str(clip.styled_subtitle_path) if clip.styled_subtitle_path else None,
                    "start": seconds_to_timecode(clip.start_seconds),
                    "end": seconds_to_timecode(clip.end_seconds),
                    "start_seconds": clip.start_seconds,
                    "end_seconds": clip.end_seconds,
                    "duration_seconds": clip.duration_seconds,
                    "hook_text": clip.hook,
                    "reason": clip.reason,
                    "viral_score": clip.score,
                }
                for clip in clips
            ],
            "edit_plans": [
                {
                    "sequence_number": index,
                    "start": seconds_to_timecode(plan.start_seconds),
                    "end": seconds_to_timecode(plan.end_seconds),
                    "duration_seconds": plan.duration_seconds,
                    "hook": plan.hook,
                    "reason": plan.reason,
                    "score": plan.score,
                    "hook_overlay": (
                        {
                            "text": plan.hook_overlay.text,
                            "start_seconds": plan.hook_overlay.start_seconds,
                            "end_seconds": plan.hook_overlay.end_seconds,
                        }
                        if plan.hook_overlay
                        else None
                    ),
                    "zoom_effects": [
                        {
                            "start_seconds": effect.start_seconds,
                            "end_seconds": effect.end_seconds,
                            "peak_scale": effect.peak_scale,
                            "anchor_text": effect.anchor_text,
                        }
                        for effect in plan.zoom_effects
                    ],
                }
                for index, plan in enumerate(edit_plans, start=1)
            ],
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return manifest_path
