from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.config import AppSettings
from src.core.models import SentenceBlock, TranscriptArtifact, TranscriptSegment

SENTENCE_END_PATTERN = re.compile(r"[.!?]['\"\)\]]*$")
TOKEN_PATTERN = re.compile(r"\S+")


@dataclass(slots=True)
class _TimedToken:
    text: str
    start_seconds: float
    end_seconds: float
    segment_index: int


class TranscriptSegmenter:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def segment(self, transcript: TranscriptArtifact) -> list[SentenceBlock]:
        ordered_segments = sorted(transcript.segments, key=lambda segment: segment.start_seconds)
        tokens = self._build_timed_tokens(ordered_segments)
        if not tokens:
            transcript.sentence_blocks = self._fallback_blocks(ordered_segments)
            return transcript.sentence_blocks

        blocks: list[SentenceBlock] = []
        current_tokens: list[_TimedToken] = []

        for token in tokens:
            if current_tokens:
                gap = token.start_seconds - current_tokens[-1].end_seconds
                if gap > self.settings.transcript_sentence_pause_seconds:
                    blocks.append(self._build_block(current_tokens))
                    current_tokens = []

            current_tokens.append(token)
            if SENTENCE_END_PATTERN.search(token.text.strip()):
                blocks.append(self._build_block(current_tokens))
                current_tokens = []

        if current_tokens:
            blocks.append(self._build_block(current_tokens))

        transcript.sentence_blocks = blocks or self._fallback_blocks(ordered_segments)
        return transcript.sentence_blocks

    def _build_timed_tokens(self, segments: list[TranscriptSegment]) -> list[_TimedToken]:
        tokens: list[_TimedToken] = []
        for segment_index, segment in enumerate(segments):
            if segment.words:
                for word in segment.words:
                    text = word.text.strip()
                    if not text:
                        continue
                    tokens.append(
                        _TimedToken(
                            text=text,
                            start_seconds=word.start_seconds,
                            end_seconds=max(word.end_seconds, word.start_seconds + 0.01),
                            segment_index=segment_index,
                        )
                    )
                continue

            segment_tokens = [match.group(0) for match in TOKEN_PATTERN.finditer(segment.text)]
            if not segment_tokens:
                continue

            duration = max(segment.end_seconds - segment.start_seconds, 0.2)
            slice_duration = duration / len(segment_tokens)
            cursor = segment.start_seconds
            for index, token in enumerate(segment_tokens):
                end_seconds = segment.end_seconds if index == len(segment_tokens) - 1 else cursor + slice_duration
                tokens.append(
                    _TimedToken(
                        text=token,
                        start_seconds=cursor,
                        end_seconds=max(end_seconds, cursor + 0.01),
                        segment_index=segment_index,
                    )
                )
                cursor = end_seconds

        return tokens

    @staticmethod
    def _build_block(tokens: list[_TimedToken]) -> SentenceBlock:
        cleaned_text = " ".join(token.text.strip() for token in tokens if token.text.strip())
        cleaned_text = re.sub(r"\s+([,.;:!?])", r"\1", cleaned_text).strip()
        return SentenceBlock(
            start_seconds=tokens[0].start_seconds,
            end_seconds=tokens[-1].end_seconds,
            text=cleaned_text,
            speaker="speaker_1",
            source_segment_start_index=tokens[0].segment_index,
            source_segment_end_index=tokens[-1].segment_index,
        )

    @staticmethod
    def _fallback_blocks(segments: list[TranscriptSegment]) -> list[SentenceBlock]:
        return [
            SentenceBlock(
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                text=segment.text.strip(),
                speaker="speaker_1",
                source_segment_start_index=index,
                source_segment_end_index=index,
            )
            for index, segment in enumerate(segments)
            if segment.text.strip()
        ]
