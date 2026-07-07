from pathlib import Path

from src.core.config import AppSettings
from src.core.models import EditPlan, HookOverlay, SpeakerFocusPoint, SpeakerFocusTrack, ZoomEffect
from src.modules.clip_generator.service import ClipGenerator


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


class CapturingRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def run(self, command: list[str], cwd: Path | None = None, on_output=None):  # noqa: ANN001,ANN201
        self.commands.append(command)
        if on_output is not None:
            on_output("out_time=00:00:01.000000")
        output_path = Path(command[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("placeholder", encoding="utf-8")
        return None


def test_clip_generator_uses_full_screen_vertical_composition_and_caption_burn_in(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings.ensure_directories()
    runner = CapturingRunner()
    generator = ClipGenerator(settings, runner)  # type: ignore[arg-type]
    generator._subtitle_filter_supported = True

    source_video = settings.videos_dir / "source.mp4"
    source_video.write_text("video", encoding="utf-8")

    clips = generator.generate(
        job_id="job-1",
        video_id="video-1",
        video_path=source_video,
        plans=[
            EditPlan(
                start_seconds=5.0,
                end_seconds=28.0,
                score=0.9,
                hook="Hook",
                reason="Reason",
                hook_overlay=HookOverlay(text="HOOK", start_seconds=0.0, end_seconds=1.5),
                zoom_effects=[
                    ZoomEffect(start_seconds=0.8, end_seconds=1.4, peak_scale=1.14, anchor_text="HOOK"),
                    ZoomEffect(start_seconds=4.0, end_seconds=4.6, peak_scale=1.10, anchor_text="POINT"),
                ],
            )
        ],
    )

    styled_subtitle = settings.clips_dir / "video-1" / "subtitles" / "styled" / "clip_001.ass"
    styled_subtitle.parent.mkdir(parents=True, exist_ok=True)
    styled_subtitle.write_text("[Events]\n", encoding="utf-8")
    burn_result = generator.burn_subtitles(clips, {1: styled_subtitle})

    render_command = runner.commands[0]
    burn_command = runner.commands[1]

    assert "-filter_complex" in render_command
    render_filter = render_command[render_command.index("-filter_complex") + 1]
    assert "scale=1080:1920:force_original_aspect_ratio=increase" in render_filter
    assert "crop=1080:1920" in render_filter
    assert "fps=30" in render_filter
    assert "setsar=1,unsharp=5:5:0.48:5:5:0.0,format=yuv420p[outv]" in render_filter
    assert "boxblur=18:8" not in render_filter
    assert "sin(((t-" not in render_filter
    assert render_command[render_command.index("-preset") + 1] == "fast"
    assert render_command[render_command.index("-b:v") + 1] == "6M"
    assert render_command[render_command.index("-maxrate") + 1] == "8M"
    assert render_command[render_command.index("-bufsize") + 1] == "12M"
    assert render_command[render_command.index("-b:a") + 1] == "320k"
    assert "-vf" in burn_command
    assert burn_result.burned_count == 1
    assert burn_result.warning_message is None
    assert "subtitles=filename=" in burn_command[burn_command.index("-vf") + 1]


def test_clip_generator_skips_caption_burn_in_when_ffmpeg_lacks_subtitle_filter(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings.ensure_directories()
    runner = CapturingRunner()
    generator = ClipGenerator(settings, runner)  # type: ignore[arg-type]
    generator._subtitle_filter_supported = False

    source_video = settings.videos_dir / "source.mp4"
    source_video.write_text("video", encoding="utf-8")

    clips = generator.generate(
        job_id="job-1",
        video_id="video-1",
        video_path=source_video,
        plans=[
            EditPlan(
                start_seconds=0.0,
                end_seconds=5.0,
                score=0.8,
                hook="Hook",
                reason="Reason",
                hook_overlay=None,
                zoom_effects=[],
            )
        ],
    )

    styled_subtitle = settings.clips_dir / "video-1" / "subtitles" / "styled" / "clip_001.ass"
    styled_subtitle.parent.mkdir(parents=True, exist_ok=True)
    styled_subtitle.write_text("[Events]\n", encoding="utf-8")

    result = generator.burn_subtitles(clips, {1: styled_subtitle})

    assert result.burned_count == 0
    assert result.warning_message is not None
    assert len(runner.commands) == 1


def test_clip_generator_supports_landscape_output_ratio(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings.ensure_directories()
    runner = CapturingRunner()
    generator = ClipGenerator(settings, runner)  # type: ignore[arg-type]

    source_video = settings.videos_dir / "source.mp4"
    source_video.write_text("video", encoding="utf-8")

    generator.generate(
        job_id="job-1",
        video_id="video-1",
        video_path=source_video,
        plans=[
            EditPlan(
                start_seconds=0.0,
                end_seconds=6.0,
                score=0.8,
                hook="Hook",
                reason="Reason",
                hook_overlay=None,
                zoom_effects=[],
            )
        ],
        output_aspect_ratio="16:9",
    )

    render_filter = runner.commands[0][runner.commands[0].index("-filter_complex") + 1]
    assert "scale=1920:1080:force_original_aspect_ratio=increase" in render_filter
    assert "crop=1920:1080" in render_filter
    assert "fps=30" in render_filter


def test_clip_generator_uses_speaker_focus_track_for_dynamic_crop(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings.ensure_directories()
    runner = CapturingRunner()
    generator = ClipGenerator(settings, runner)  # type: ignore[arg-type]

    source_video = settings.videos_dir / "source.mp4"
    source_video.write_text("video", encoding="utf-8")

    generator.generate(
        job_id="job-1",
        video_id="video-1",
        video_path=source_video,
        plans=[
            EditPlan(
                start_seconds=0.0,
                end_seconds=6.0,
                score=0.8,
                hook="Hook",
                reason="Reason",
                hook_overlay=None,
                zoom_effects=[],
            )
        ],
        focus_tracks={
            1: SpeakerFocusTrack(
                source_width=1920,
                source_height=1080,
                points=[
                    SpeakerFocusPoint(time_seconds=0.0, center_x=0.28, center_y=0.44),
                    SpeakerFocusPoint(time_seconds=3.0, center_x=0.72, center_y=0.46),
                    SpeakerFocusPoint(time_seconds=6.0, center_x=0.66, center_y=0.45),
                ],
            )
        },
    )

    render_filter = runner.commands[0][runner.commands[0].index("-filter_complex") + 1]
    assert "crop=w=608:h=1080" in render_filter
    assert "x='if(lt(t,3.000)" in render_filter
    assert "scale=1080:1920:flags=lanczos" in render_filter
    assert "fps=30" in render_filter


def test_clip_generator_simplifies_long_focus_tracks_before_building_expression(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings.ensure_directories()
    runner = CapturingRunner()
    generator = ClipGenerator(settings, runner)  # type: ignore[arg-type]

    source_video = settings.videos_dir / "source.mp4"
    source_video.write_text("video", encoding="utf-8")

    points = [
        SpeakerFocusPoint(
            time_seconds=round(index * 0.45, 3),
            center_x=0.2 + ((index % 9) * 0.06),
            center_y=0.45,
        )
        for index in range(180)
    ]

    generator.generate(
        job_id="job-1",
        video_id="video-1",
        video_path=source_video,
        plans=[
            EditPlan(
                start_seconds=0.0,
                end_seconds=80.0,
                score=0.8,
                hook="Hook",
                reason="Reason",
                hook_overlay=None,
                zoom_effects=[],
            )
        ],
        focus_tracks={
            1: SpeakerFocusTrack(
                source_width=1920,
                source_height=1080,
                points=points,
            )
        },
    )

    render_filter = runner.commands[0][runner.commands[0].index("-filter_complex") + 1]
    assert render_filter.count("if(lt(t,") <= 48
