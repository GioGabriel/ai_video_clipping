from pathlib import Path

from src.core.config import AppSettings
from src.core.models import TranscriptArtifact, TranscriptSegment, TranscriptWord, ViralMoment
from src.modules.edit_planner.service import EditPlanner


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
        min_clip_duration_seconds=12,
        max_clip_duration_seconds=75,
        target_clip_duration_seconds=30,
        viral_chunk_duration_seconds=420,
        viral_chunk_overlap_seconds=60,
        viral_min_score=60,
        subtitle_generation_enabled=True,
        edit_plan_zoom_enabled=True,
        edit_plan_max_zoom_beats=3,
    )


def test_edit_planner_builds_zoom_beats_and_hook_overlay(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    planner = EditPlanner(settings)

    transcript = TranscriptArtifact(
        video_id="video-1",
        language="en",
        text="This productivity trick is insane and it changes everything when you do it daily.",
        transcript_path=settings.transcripts_dir / "video-1.json",
        segments=[
            TranscriptSegment(
                start_seconds=10.0,
                end_seconds=18.0,
                text="This productivity trick is insane and it changes everything when you do it daily.",
                words=[
                    TranscriptWord(10.0, 10.4, "This"),
                    TranscriptWord(10.4, 10.8, "productivity"),
                    TranscriptWord(10.8, 11.1, "trick"),
                    TranscriptWord(11.1, 11.5, "is"),
                    TranscriptWord(11.5, 11.9, "insane"),
                    TranscriptWord(11.9, 12.3, "and"),
                    TranscriptWord(12.3, 12.7, "it"),
                    TranscriptWord(12.7, 13.2, "changes"),
                    TranscriptWord(13.2, 13.8, "everything"),
                    TranscriptWord(13.8, 14.2, "when"),
                    TranscriptWord(14.2, 14.6, "you"),
                    TranscriptWord(14.6, 15.0, "do"),
                    TranscriptWord(15.0, 15.4, "it"),
                    TranscriptWord(15.4, 16.1, "daily"),
                ],
            )
        ],
    )

    plans = planner.plan(
        transcript,
        [
            ViralMoment(
                start_seconds=10.0,
                end_seconds=31.5,
                score=0.92,
                hook="This productivity trick is insane",
                reason="Strong opinion and advice",
            )
        ],
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.duration_seconds > 20
    assert plan.hook_overlay is not None
    assert "productivity" in plan.hook_overlay.text.lower()
    assert plan.hook_overlay.text != plan.hook_overlay.text.upper()
    assert len(plan.zoom_effects) >= 1
    assert any(effect.anchor_text and "insane" in effect.anchor_text.lower() for effect in plan.zoom_effects)
