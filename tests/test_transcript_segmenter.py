from pathlib import Path

from src.core.config import AppSettings
from src.core.models import TranscriptArtifact, TranscriptSegment, TranscriptWord
from src.modules.transcript_segmenter.service import TranscriptSegmenter


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
        min_clip_duration_seconds=18,
        max_clip_duration_seconds=50,
        target_clip_duration_seconds=30,
        viral_chunk_duration_seconds=420,
        viral_chunk_overlap_seconds=60,
        viral_min_score=60,
        subtitle_generation_enabled=True,
    )


def test_transcript_segmenter_builds_sentence_blocks_from_word_timestamps(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    segmenter = TranscriptSegmenter(settings)
    transcript = TranscriptArtifact(
        video_id="video-1",
        language="en",
        text="Why does discipline equal freedom? Because the more discipline you have, the more freedom you get.",
        transcript_path=settings.transcripts_dir / "video-1.json",
        segments=[
            TranscriptSegment(
                start_seconds=0.0,
                end_seconds=7.0,
                text="Why does discipline equal freedom? Because the more discipline you have, the more freedom you get.",
                words=[
                    TranscriptWord(0.0, 0.3, "Why"),
                    TranscriptWord(0.3, 0.6, "does"),
                    TranscriptWord(0.6, 1.2, "discipline"),
                    TranscriptWord(1.2, 1.5, "equal"),
                    TranscriptWord(1.5, 2.2, "freedom?"),
                    TranscriptWord(2.5, 3.0, "Because"),
                    TranscriptWord(3.0, 3.3, "the"),
                    TranscriptWord(3.3, 3.7, "more"),
                    TranscriptWord(3.7, 4.5, "discipline"),
                    TranscriptWord(4.5, 4.8, "you"),
                    TranscriptWord(4.8, 5.1, "have,"),
                    TranscriptWord(5.1, 5.4, "the"),
                    TranscriptWord(5.4, 5.8, "more"),
                    TranscriptWord(5.8, 6.5, "freedom"),
                    TranscriptWord(6.5, 7.0, "you get."),
                ],
            )
        ],
    )

    blocks = segmenter.segment(transcript)

    assert len(blocks) == 2
    assert blocks[0].text == "Why does discipline equal freedom?"
    assert blocks[1].text.startswith("Because the more discipline")


def test_transcript_segmenter_splits_on_long_pause_even_without_punctuation(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings.transcript_sentence_pause_seconds = 0.8
    segmenter = TranscriptSegmenter(settings)
    transcript = TranscriptArtifact(
        video_id="video-1",
        language="en",
        text="Most people get this wrong and then they stay stuck",
        transcript_path=settings.transcripts_dir / "video-1.json",
        segments=[
            TranscriptSegment(0.0, 2.0, "Most people get this wrong"),
            TranscriptSegment(3.4, 5.2, "and then they stay stuck"),
        ],
    )

    blocks = segmenter.segment(transcript)

    assert len(blocks) == 2
