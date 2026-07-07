from pathlib import Path

from src.core.config import AppSettings
from src.core.models import ClipArtifact, EditPlan, HookOverlay, TranscriptArtifact, TranscriptSegment, TranscriptWord
from src.modules.subtitle_generator.service import SubtitleGenerator


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
        subtitle_words_per_cue=3,
        subtitle_max_chars_per_cue=18,
    )


def test_subtitle_generator_emits_srt_and_ass_files(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings.ensure_directories()
    generator = SubtitleGenerator(settings)

    transcript = TranscriptArtifact(
        video_id="video-1",
        language="en",
        text="The quick brown fox jumps over the lazy dog and keeps talking.",
        transcript_path=settings.transcripts_dir / "video-1.json",
        segments=[
            TranscriptSegment(
                start_seconds=1.0,
                end_seconds=5.0,
                text="The quick brown fox jumps over the lazy dog and keeps talking.",
                words=[
                    TranscriptWord(1.0, 1.3, "The"),
                    TranscriptWord(1.3, 1.6, "quick"),
                    TranscriptWord(1.6, 1.9, "brown"),
                    TranscriptWord(1.9, 2.2, "fox"),
                    TranscriptWord(2.2, 2.6, "jumps"),
                    TranscriptWord(2.6, 2.9, "over"),
                    TranscriptWord(2.9, 3.2, "the"),
                    TranscriptWord(3.2, 3.6, "lazy"),
                    TranscriptWord(3.6, 4.0, "dog"),
                    TranscriptWord(4.0, 4.4, "and"),
                    TranscriptWord(4.4, 4.7, "keeps"),
                    TranscriptWord(4.7, 5.0, "talking"),
                ],
            )
        ],
    )
    clip = ClipArtifact(
        job_id="job-1",
        video_id="video-1",
        sequence_number=1,
        file_path=settings.clips_dir / "video-1" / "clip_001.mp4",
        start_seconds=0.5,
        end_seconds=5.0,
        hook="Hook",
        reason="Reason",
        score=0.9,
    )

    artifacts = generator.generate(
        "video-1",
        transcript,
        [clip],
        {
            1: EditPlan(
                start_seconds=0.5,
                end_seconds=5.0,
                score=0.9,
                hook="Build this clip right",
                reason="Testing overlay",
                zoom_effects=[],
                hook_overlay=HookOverlay(text="BUILD THIS\nCLIP RIGHT", start_seconds=0.0, end_seconds=1.8),
            )
        },
    )

    subtitle_artifact = artifacts[1]
    srt_text = subtitle_artifact.sidecar_path.read_text(encoding="utf-8")
    ass_text = subtitle_artifact.styled_path.read_text(encoding="utf-8")

    assert subtitle_artifact.sidecar_path.suffix == ".srt"
    assert subtitle_artifact.styled_path.suffix == ".ass"
    assert "The quick brown" in srt_text
    assert "[V4+ Styles]" in ass_text
    assert "Dialogue: 0" in ass_text
    assert "HookTop" in ass_text
    assert "Style: HookTop,Arial Black,62" in ass_text
    assert ",120,120,160,1" in ass_text
    assert "BUILD THIS\\NCLIP RIGHT" in ass_text
    assert r"\rViralActive" in ass_text
    assert r"{\fad(90,140)}" in ass_text
    assert r"\fscx82\fscy82" not in ass_text
    assert "TALKING" in ass_text


def test_subtitle_generator_clips_sidecar_words_to_overlap_and_supports_landscape_ass(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings.ensure_directories()
    generator = SubtitleGenerator(settings)

    transcript = TranscriptArtifact(
        video_id="video-2",
        language="en",
        text="one two three four",
        transcript_path=settings.transcripts_dir / "video-2.json",
        segments=[
            TranscriptSegment(
                start_seconds=0.0,
                end_seconds=4.0,
                text="one two three four",
                words=[
                    TranscriptWord(0.0, 1.0, "one"),
                    TranscriptWord(1.0, 2.0, "two"),
                    TranscriptWord(2.0, 3.0, "three"),
                    TranscriptWord(3.0, 4.0, "four"),
                ],
            )
        ],
    )
    clip = ClipArtifact(
        job_id="job-2",
        video_id="video-2",
        sequence_number=1,
        file_path=settings.clips_dir / "video-2" / "clip_001.mp4",
        start_seconds=1.0,
        end_seconds=2.0,
        hook="Hook",
        reason="Reason",
        score=0.9,
    )

    artifacts = generator.generate(
        "video-2",
        transcript,
        [clip],
        output_aspect_ratio="16:9",
        caption_theme="cinematic",
    )

    srt_text = artifacts[1].sidecar_path.read_text(encoding="utf-8")
    ass_text = artifacts[1].styled_path.read_text(encoding="utf-8")

    assert "one two three four" not in srt_text
    assert "\ntwo\n" in srt_text.lower()
    assert "PlayResX: 1920" in ass_text
    assert "PlayResY: 1080" in ass_text


def test_subtitle_generator_keeps_each_overlay_word_on_its_own_timestamp(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    generator = SubtitleGenerator(settings)

    transcript = TranscriptArtifact(
        video_id="video-3",
        language="en",
        text="to go now",
        transcript_path=tmp_path / "video-3.json",
        segments=[
            TranscriptSegment(
                start_seconds=0.0,
                end_seconds=1.5,
                text="to go now",
                words=[
                    TranscriptWord(0.0, 0.2, "to"),
                    TranscriptWord(0.2, 0.6, "go"),
                    TranscriptWord(0.6, 1.0, "now"),
                ],
            )
        ],
    )
    clip = ClipArtifact(
        job_id="job-3",
        video_id="video-3",
        sequence_number=1,
        file_path=tmp_path / "clip_001.mp4",
        start_seconds=0.0,
        end_seconds=1.0,
        hook="Hook",
        reason="Reason",
        score=0.9,
    )

    cues = generator.build_karaoke_cues(clip, transcript, caption_theme="tiktok")

    assert [cue.spoken_text for cue in cues] == ["TO", "GO", "NOW"]
    assert cues[0].start_seconds == 0.0
    assert cues[1].start_seconds == 0.2
    assert cues[2].start_seconds == 0.6
