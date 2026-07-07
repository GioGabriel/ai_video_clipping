from __future__ import annotations

import re

from src.core.config import AppSettings
from src.core.models import SentenceBlock, ViralMoment

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "have",
    "i",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "we",
    "with",
    "you",
}
PAYOFF_TERMS = ("because", "therefore", "that is why", "so", "which means", "you will", "freedom", "truth")


class ThoughtCompletion:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def complete(self, sentence_blocks: list[SentenceBlock], moments: list[ViralMoment]) -> list[ViralMoment]:
        if not sentence_blocks:
            return moments
        return [self._complete_for_moment(sentence_blocks, moment) for moment in moments]

    def _complete_for_moment(self, sentence_blocks: list[SentenceBlock], moment: ViralMoment) -> ViralMoment:
        start_time = moment.start_seconds
        core_end = moment.core_end_seconds or moment.end_seconds
        start_index = self._find_block_index(sentence_blocks, start_time)
        end_index = self._find_block_index(sentence_blocks, core_end)
        if start_index is None or end_index is None:
            return moment

        max_end_time = start_time + max(self.settings.max_clip_duration_seconds, self.settings.target_clip_duration_seconds)
        while end_index + 1 < len(sentence_blocks):
            current = sentence_blocks[end_index]
            following = sentence_blocks[end_index + 1]
            if following.start_seconds > max_end_time:
                break
            if following.speaker != current.speaker:
                break
            gap = following.start_seconds - current.end_seconds
            current_duration = following.end_seconds - start_time
            if gap > self.settings.thought_completion_pause_seconds and current_duration >= self.settings.min_clip_duration_seconds:
                break
            if not self._blocks_belong_together(current, following) and current_duration >= self.settings.target_clip_duration_seconds:
                break
            end_index += 1

        final_end = min(
            sentence_blocks[end_index].end_seconds + self.settings.clip_trailing_pad_seconds,
            start_time + self.settings.max_clip_duration_seconds,
        )
        return ViralMoment(
            start_seconds=moment.start_seconds,
            end_seconds=max(final_end, moment.end_seconds),
            score=moment.score,
            hook=moment.hook,
            reason=moment.reason,
            hook_start_seconds=moment.hook_start_seconds,
            core_start_seconds=moment.core_start_seconds or moment.start_seconds,
            core_end_seconds=sentence_blocks[end_index].end_seconds,
            hook_strength=moment.hook_strength,
            emotion_level=moment.emotion_level,
            statement_strength=moment.statement_strength,
            novelty=moment.novelty,
            duration_score=moment.duration_score,
            phrase_score=moment.phrase_score,
        )

    @staticmethod
    def _find_block_index(sentence_blocks: list[SentenceBlock], timestamp: float) -> int | None:
        for index, block in enumerate(sentence_blocks):
            if block.start_seconds <= timestamp <= block.end_seconds:
                return index
        for index, block in enumerate(sentence_blocks):
            if block.start_seconds >= timestamp:
                return index
        return len(sentence_blocks) - 1 if sentence_blocks else None

    def _blocks_belong_together(self, left: SentenceBlock, right: SentenceBlock) -> bool:
        if any(term in left.text.lower() for term in PAYOFF_TERMS) and self._share_theme(left.text, right.text):
            return True
        gap = right.start_seconds - left.end_seconds
        if gap <= self.settings.clip_sentence_gap_threshold_seconds and self._share_theme(left.text, right.text):
            return True
        return gap <= self.settings.thought_completion_pause_seconds and self._share_theme(left.text, right.text)

    @staticmethod
    def _share_theme(left_text: str, right_text: str) -> bool:
        left_tokens = {
            token.lower()
            for token in TOKEN_PATTERN.findall(left_text)
            if token.lower() not in STOPWORDS and len(token) > 2
        }
        right_tokens = {
            token.lower()
            for token in TOKEN_PATTERN.findall(right_text)
            if token.lower() not in STOPWORDS and len(token) > 2
        }
        return bool(left_tokens.intersection(right_tokens))
