from pathlib import Path

from src.core.config import AppSettings
from src.core.models import TranscriptSegment, ViralMoment
from src.modules.moment_refiner.service import MomentRefiner


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
        max_clip_duration_seconds=20,
        target_clip_duration_seconds=30,
        viral_chunk_duration_seconds=420,
        viral_chunk_overlap_seconds=60,
        viral_min_score=60,
        subtitle_generation_enabled=True,
        clip_lead_in_seconds=0.5,
        clip_trailing_pad_seconds=0.8,
        clip_sentence_gap_threshold_seconds=1.0,
    )


def test_refiner_extends_clip_to_finish_sentence(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    refiner = MomentRefiner(settings)

    refined = refiner.refine(
        segments=[
            TranscriptSegment(10.0, 13.0, "This is the setup"),
            TranscriptSegment(13.0, 17.0, "and this is the viral line"),
            TranscriptSegment(17.0, 20.2, "that lands right here."),
            TranscriptSegment(21.8, 24.0, "New sentence after a pause."),
        ],
        moments=[
            ViralMoment(
                start_seconds=12.8,
                end_seconds=17.1,
                score=0.9,
                hook="viral line",
                reason="Strong payoff",
            )
        ],
    )

    assert refined[0].start_seconds <= 12.3
    assert refined[0].end_seconds >= 21.0
    assert refined[0].end_seconds < 21.8


def test_refiner_allows_longer_thoughts_to_finish(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    refiner = MomentRefiner(settings)

    refined = refiner.refine(
        segments=[
            TranscriptSegment(0.0, 8.0, "A very long run-on statement"),
            TranscriptSegment(8.0, 16.0, "that keeps going without punctuation"),
            TranscriptSegment(16.0, 24.0, "and still does not stop"),
            TranscriptSegment(24.0, 31.0, "until much later finally."),
        ],
        moments=[
            ViralMoment(
                start_seconds=2.0,
                end_seconds=18.0,
                score=0.8,
                hook="run on",
                reason="Test grace limit",
            )
        ],
    )

    assert refined[0].end_seconds >= 31.0


def test_refiner_can_cross_soft_pause_when_sentence_has_not_finished(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    refiner = MomentRefiner(settings)

    refined = refiner.refine(
        segments=[
            TranscriptSegment(30.0, 33.0, "This point is important and"),
            TranscriptSegment(34.2, 37.5, "it keeps going after a short pause"),
            TranscriptSegment(37.5, 40.0, "until it finally lands."),
        ],
        moments=[
            ViralMoment(
                start_seconds=30.5,
                end_seconds=33.1,
                score=0.75,
                hook="important point",
                reason="Pause continuation",
            )
        ],
    )

    assert refined[0].end_seconds >= 40.5


def test_refiner_keeps_quotable_reaction_block_together(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    refiner = MomentRefiner(settings)

    refined = refiner.refine(
        segments=[
            TranscriptSegment(0.0, 4.5, "Everyone is jealous of what you've got. No one is jealous of how you got it."),
            TranscriptSegment(4.5, 6.4, "I love that quote. My God."),
            TranscriptSegment(6.4, 11.2, "People see the trophies, but not the training ground. Everybody wants the view, but no one wants the climb."),
            TranscriptSegment(13.5, 17.0, "Now we are changing topics."),
        ],
        moments=[
            ViralMoment(
                start_seconds=4.8,
                end_seconds=6.0,
                score=0.93,
                hook="No one is jealous of how you got it",
                reason="Contrast-driven aphorism",
            )
        ],
    )

    assert refined[0].start_seconds <= 0.0
    assert refined[0].end_seconds >= 12.0
    assert refined[0].end_seconds < 13.5


def test_refiner_rewinds_to_earlier_motivational_thesis_and_finishes_payoff(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    refiner = MomentRefiner(settings)

    refined = refiner.refine(
        segments=[
            TranscriptSegment(0.0, 4.0, "Why does discipline equal freedom?"),
            TranscriptSegment(4.0, 9.0, "Because the more discipline you have in your life, the more freedom you will end up with."),
            TranscriptSegment(9.0, 14.0, "If you lack the discipline to manage your time, you will end up with no free time."),
            TranscriptSegment(14.0, 19.0, "If you have discipline, you will attain freedom."),
            TranscriptSegment(19.0, 24.0, "And if you do that, you're going to end up with freedom across the board."),
            TranscriptSegment(26.5, 30.0, "Now we are changing topics."),
        ],
        moments=[
            ViralMoment(
                start_seconds=18.7,
                end_seconds=22.0,
                score=0.88,
                hook="you're going to end up with freedom",
                reason="Late summary fragment",
            )
        ],
    )

    assert refined[0].start_seconds <= 0.0
    assert refined[0].end_seconds >= 24.5
    assert refined[0].end_seconds < 26.5
