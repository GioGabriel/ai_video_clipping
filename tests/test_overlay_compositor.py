from pathlib import Path

from src.core.config import AppSettings
from src.core.models import ClipArtifact, EditPlan, HookOverlay, TranscriptArtifact, TranscriptSegment, TranscriptWord
from src.modules.overlay_compositor.service import OverlayCompositor
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
        min_clip_duration_seconds=12,
        max_clip_duration_seconds=120,
        target_clip_duration_seconds=30,
        viral_chunk_duration_seconds=420,
        viral_chunk_overlap_seconds=60,
        viral_min_score=60,
        subtitle_generation_enabled=True,
        subtitle_font_size=72,
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
        output_path.write_text("captioned", encoding="utf-8")
        return None


def test_overlay_compositor_burns_local_karaoke_overlays(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings.ensure_directories()
    subtitle_generator = SubtitleGenerator(settings)
    runner = CapturingRunner()
    compositor = OverlayCompositor(settings, runner, subtitle_generator)  # type: ignore[arg-type]

    clip_path = settings.clips_dir / "video-1" / "clip_001.mp4"
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.write_text("base", encoding="utf-8")

    transcript = TranscriptArtifact(
        video_id="video-1",
        language="en",
        text="Discipline equals freedom.",
        transcript_path=settings.transcripts_dir / "video-1.json",
        segments=[
            TranscriptSegment(
                start_seconds=0.0,
                end_seconds=2.4,
                text="Discipline equals freedom.",
                words=[
                    TranscriptWord(start_seconds=0.0, end_seconds=0.7, text="Discipline"),
                    TranscriptWord(start_seconds=0.7, end_seconds=1.4, text="equals"),
                    TranscriptWord(start_seconds=1.4, end_seconds=2.4, text="freedom."),
                ],
            )
        ],
    )
    clip = ClipArtifact(
        job_id="job-1",
        video_id="video-1",
        sequence_number=1,
        file_path=clip_path,
        start_seconds=0.0,
        end_seconds=2.4,
        hook="Discipline equals freedom",
        reason="Strong hook",
        score=0.98,
    )
    plan = EditPlan(
        start_seconds=0.0,
        end_seconds=2.4,
        score=0.98,
        hook="Discipline equals freedom",
        reason="Strong hook",
        zoom_effects=[],
        hook_overlay=HookOverlay(text="DISCIPLINE\nEQUALS FREEDOM", start_seconds=0.0, end_seconds=1.6),
    )

    result = compositor.apply(
        video_id="video-1",
        transcript=transcript,
        clips=[clip],
        edit_plans={1: plan},
    )

    assert result.burned_count == 1
    assert clip_path.read_text(encoding="utf-8") == "captioned"
    assert len(runner.commands) == 1
    command = runner.commands[0]
    assert "-f" in command
    assert command[command.index("-f") + 1] == "concat"
    assert "overlay=0:0:eof_action=pass" in command[command.index("-filter_complex") + 1]
    assert command[command.index("-preset") + 1] == "fast"
    assert command[command.index("-b:v") + 1] == "6M"
    assert command[command.index("-c:a") + 1] == "copy"
