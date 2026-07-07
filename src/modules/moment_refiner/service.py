from __future__ import annotations

import re

from src.core.config import AppSettings
from src.core.models import TranscriptSegment, ViralMoment

SENTENCE_END_PATTERN = re.compile(r"[.!?]['\"\)\]]*$")
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
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "their",
    "there",
    "they",
    "this",
    "to",
    "we",
    "what",
    "when",
    "with",
    "you",
    "your",
}
CONTINUATION_PREFIXES = (
    "and ",
    "but ",
    "because ",
    "so ",
    "so if ",
    "and if ",
    "that's why ",
    "this is why ",
    "or ",
    "then ",
    "which ",
    "that ",
    "if ",
    "when ",
    "everyone ",
    "everybody ",
    "people ",
    "no one ",
    "nobody ",
    "the reason ",
    "this is ",
    "it is ",
    "it doesn't ",
    "it does ",
    "discipline ",
    "motivation ",
    "action ",
    "don't ",
    "do not ",
    "just ",
)
REACTION_PHRASES = (
    "i love",
    "my god",
    "wow",
    "holy",
    "that's crazy",
    "that is crazy",
    "that quote",
)
PARALLEL_QUOTE_PATTERNS = (
    re.compile(r"\beveryone\b.*\bno one\b"),
    re.compile(r"\beverybody\b.*\bno one\b"),
    re.compile(r"\bpeople\b.*\bbut\b.*\bnot\b"),
    re.compile(r"\bwinners\b.*\blosers\b"),
    re.compile(r"\byou\b.*\bbut\b.*\byou\b"),
)
QUESTION_START_PATTERN = re.compile(r"^(why|how|what|when|are|do|does|did|can|should|could)\b", re.IGNORECASE)
MOTIVATIONAL_PATTERNS = (
    re.compile(r"\bdiscipline\b.*\bfreedom\b", re.IGNORECASE),
    re.compile(r"\bextreme ownership\b", re.IGNORECASE),
    re.compile(r"\bmotivation\b.*\bdiscipline\b", re.IGNORECASE),
    re.compile(r"\bif you\b.*\byou will\b", re.IGNORECASE),
    re.compile(r"\bslave to\b", re.IGNORECASE),
    re.compile(r"\bget up\b", re.IGNORECASE),
    re.compile(r"\bmove towards\b", re.IGNORECASE),
    re.compile(r"\btake action\b", re.IGNORECASE),
    re.compile(r"\bdon't\b.*\btoday\b", re.IGNORECASE),
)
PAYOFF_PATTERNS = (
    re.compile(r"\bbecause\b", re.IGNORECASE),
    re.compile(r"\bthat's why\b", re.IGNORECASE),
    re.compile(r"\bthis is why\b", re.IGNORECASE),
    re.compile(r"\byou will\b", re.IGNORECASE),
    re.compile(r"\bend up\b", re.IGNORECASE),
    re.compile(r"\bso that\b", re.IGNORECASE),
    re.compile(r"\bthen\b", re.IGNORECASE),
    re.compile(r"\bfreedom\b", re.IGNORECASE),
    re.compile(r"\bownership\b", re.IGNORECASE),
)


class MomentRefiner:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def refine(
        self,
        segments: list[TranscriptSegment],
        moments: list[ViralMoment],
    ) -> list[ViralMoment]:
        ordered_segments = sorted(segments, key=lambda segment: segment.start_seconds)
        if not ordered_segments:
            return moments

        thought_blocks = self._build_thought_blocks(ordered_segments)
        return [self._refine_moment(moment, ordered_segments, thought_blocks) for moment in moments]

    def _refine_moment(
        self,
        moment: ViralMoment,
        segments: list[TranscriptSegment],
        thought_blocks: list[tuple[int, int]],
    ) -> ViralMoment:
        start_index = self._find_segment_index(segments, moment.start_seconds)
        end_index = self._find_segment_index(segments, moment.end_seconds)

        if start_index is None or end_index is None:
            return moment

        start_block_index = self._find_block_index(thought_blocks, start_index)
        end_block_index = self._find_block_index(thought_blocks, end_index)
        start_block_index, end_block_index = self._expand_related_blocks(
            segments=segments,
            thought_blocks=thought_blocks,
            start_block_index=start_block_index,
            end_block_index=end_block_index,
            moment=moment,
        )
        block_start_index = thought_blocks[start_block_index][0]
        block_end_index = thought_blocks[end_block_index][1]

        max_allowed_end = moment.start_seconds + self.settings.clip_thought_max_duration_seconds
        while (
            block_end_index > end_index
            and segments[block_end_index].end_seconds + self.settings.clip_trailing_pad_seconds > max_allowed_end
        ):
            block_end_index -= 1

        refined_start = min(
            max(moment.start_seconds - self.settings.clip_lead_in_seconds, 0.0),
            segments[block_start_index].start_seconds,
        )
        refined_end = max(
            segments[block_end_index].end_seconds + self.settings.clip_trailing_pad_seconds,
            moment.end_seconds + self.settings.clip_trailing_pad_seconds,
        )

        return ViralMoment(
            start_seconds=max(refined_start, 0.0),
            end_seconds=max(refined_end, refined_start + 1.0),
            score=moment.score,
            hook=moment.hook,
            reason=moment.reason,
            phrase_score=moment.phrase_score,
        )

    def _expand_related_blocks(
        self,
        segments: list[TranscriptSegment],
        thought_blocks: list[tuple[int, int]],
        start_block_index: int,
        end_block_index: int,
        moment: ViralMoment,
    ) -> tuple[int, int]:
        hook_tokens = self._content_tokens(f"{moment.hook} {moment.reason}")
        max_duration = max(self.settings.clip_thought_max_duration_seconds, self.settings.min_clip_duration_seconds)

        while start_block_index > 0:
            previous_block = thought_blocks[start_block_index - 1]
            candidate_duration = self._block_duration(segments, thought_blocks, start_block_index - 1, end_block_index)
            if candidate_duration > max_duration:
                break
            if not self._should_expand_backward(
                segments=segments,
                previous_block=previous_block,
                current_blocks=thought_blocks[start_block_index : end_block_index + 1],
                hook_tokens=hook_tokens,
            ):
                break
            start_block_index -= 1

        while end_block_index + 1 < len(thought_blocks):
            next_block = thought_blocks[end_block_index + 1]
            candidate_duration = self._block_duration(segments, thought_blocks, start_block_index, end_block_index + 1)
            if candidate_duration > max_duration:
                break

            current_duration = self._block_duration(segments, thought_blocks, start_block_index, end_block_index)
            if not self._should_expand_forward(
                segments=segments,
                current_blocks=thought_blocks[start_block_index : end_block_index + 1],
                next_block=next_block,
                hook_tokens=hook_tokens,
                current_duration=current_duration,
            ):
                break
            end_block_index += 1

        return start_block_index, end_block_index

    def _build_thought_blocks(self, segments: list[TranscriptSegment]) -> list[tuple[int, int]]:
        blocks: list[tuple[int, int]] = []
        block_start = 0

        for index in range(1, len(segments)):
            previous = segments[index - 1]
            current = segments[index]
            if self._starts_new_block(previous, current):
                blocks.append((block_start, index - 1))
                block_start = index

        blocks.append((block_start, len(segments) - 1))
        return blocks

    def _starts_new_block(self, previous: TranscriptSegment, current: TranscriptSegment) -> bool:
        gap_seconds = current.start_seconds - previous.end_seconds
        if gap_seconds > self.settings.clip_sentence_hard_gap_threshold_seconds:
            return True

        if self._segments_are_linked(previous, current):
            return False

        if gap_seconds > self.settings.clip_sentence_gap_threshold_seconds:
            return True

        return bool(SENTENCE_END_PATTERN.search(previous.text.strip()))

    def _segments_are_linked(self, left: TranscriptSegment, right: TranscriptSegment) -> bool:
        gap_seconds = right.start_seconds - left.end_seconds
        if gap_seconds > self.settings.clip_sentence_hard_gap_threshold_seconds:
            return False

        left_text = left.text.strip()
        right_text = right.text.strip()
        if not left_text or not right_text:
            return False

        if not SENTENCE_END_PATTERN.search(left_text):
            return True

        lowered_right = right_text.lower()
        if lowered_right.startswith(CONTINUATION_PREFIXES):
            return True

        if self._is_reaction_line(left_text) or self._is_reaction_line(right_text):
            return gap_seconds <= self.settings.clip_sentence_gap_threshold_seconds * 1.4

        if self._has_parallel_quote_pattern(left_text) and self._has_parallel_quote_pattern(right_text):
            return True

        return self._shares_theme(left_text, right_text)

    def _shares_theme(self, left_text: str, right_text: str) -> bool:
        left_tokens = self._content_tokens(left_text)
        right_tokens = self._content_tokens(right_text)
        if left_tokens.intersection(right_tokens):
            return True

        if self._has_parallel_quote_pattern(left_text) and self._is_reaction_line(right_text):
            return True

        if self._is_reaction_line(left_text) and self._has_parallel_quote_pattern(right_text):
            return True

        if self._is_motivational_text(left_text) and self._is_motivational_text(right_text):
            return True

        return False

    def _should_expand_backward(
        self,
        segments: list[TranscriptSegment],
        previous_block: tuple[int, int],
        current_blocks: list[tuple[int, int]],
        hook_tokens: set[str],
    ) -> bool:
        previous_text = self._block_text(segments, previous_block)
        current_text = self._combined_block_text(segments, current_blocks)
        if not previous_text or not current_text:
            return False

        previous_tokens = self._content_tokens(previous_text)
        current_tokens = self._content_tokens(current_text)
        shared_tokens = previous_tokens.intersection(current_tokens)
        shared_hook_tokens = previous_tokens.intersection(hook_tokens)
        gap_seconds = segments[current_blocks[0][0]].start_seconds - segments[previous_block[1]].end_seconds

        if QUESTION_START_PATTERN.search(previous_text.strip()) and current_text.lower().lstrip().startswith(("because", "and because")):
            return True
        if shared_hook_tokens:
            return True
        if len(shared_tokens) >= 2 and gap_seconds <= self.settings.clip_sentence_hard_gap_threshold_seconds:
            return True
        if self._has_parallel_quote_pattern(previous_text) and (
            self._has_parallel_quote_pattern(current_text)
            or self._is_reaction_line(current_text)
        ):
            return True
        if self._is_motivational_text(previous_text) and (
            self._is_motivational_text(current_text)
            or len(shared_tokens) >= 1
        ):
            return True
        return False

    def _should_expand_forward(
        self,
        segments: list[TranscriptSegment],
        current_blocks: list[tuple[int, int]],
        next_block: tuple[int, int],
        hook_tokens: set[str],
        current_duration: float,
    ) -> bool:
        current_text = self._combined_block_text(segments, current_blocks)
        next_text = self._block_text(segments, next_block)
        if not next_text:
            return False

        next_tokens = self._content_tokens(next_text)
        current_tokens = self._content_tokens(current_text)
        shared_tokens = current_tokens.intersection(next_tokens)
        shared_hook_tokens = next_tokens.intersection(hook_tokens)
        gap_seconds = segments[next_block[0]].start_seconds - segments[current_blocks[-1][1]].end_seconds

        if current_duration < self.settings.min_clip_duration_seconds:
            if gap_seconds <= self.settings.clip_sentence_hard_gap_threshold_seconds and (
                shared_tokens
                or shared_hook_tokens
                or self._is_motivational_text(next_text)
                or next_text.lower().strip().startswith(CONTINUATION_PREFIXES)
                or self._carries_payoff(next_text)
            ):
                return True

        if next_text.lower().strip().startswith(CONTINUATION_PREFIXES):
            return True
        if shared_hook_tokens:
            return True
        if len(shared_tokens) >= 2:
            return True
        if self._carries_payoff(next_text) and (
            self._is_motivational_text(current_text)
            or QUESTION_START_PATTERN.search(current_text.strip())
            or self._has_parallel_quote_pattern(current_text)
        ):
            return True
        if self._is_reaction_line(next_text) and (
            self._has_parallel_quote_pattern(current_text)
            or self._is_motivational_text(current_text)
        ):
            return True
        return False

    @staticmethod
    def _block_duration(
        segments: list[TranscriptSegment],
        thought_blocks: list[tuple[int, int]],
        start_block_index: int,
        end_block_index: int,
    ) -> float:
        start_index = thought_blocks[start_block_index][0]
        end_index = thought_blocks[end_block_index][1]
        return max(segments[end_index].end_seconds - segments[start_index].start_seconds, 0.0)

    @staticmethod
    def _block_text(segments: list[TranscriptSegment], block: tuple[int, int]) -> str:
        start_index, end_index = block
        return " ".join(segment.text.strip() for segment in segments[start_index : end_index + 1] if segment.text.strip()).strip()

    def _combined_block_text(self, segments: list[TranscriptSegment], blocks: list[tuple[int, int]]) -> str:
        if not blocks:
            return ""
        parts = [self._block_text(segments, block) for block in blocks]
        return " ".join(part for part in parts if part).strip()

    @staticmethod
    def _find_segment_index(segments: list[TranscriptSegment], second: float) -> int | None:
        for index, segment in enumerate(segments):
            if segment.start_seconds <= second <= segment.end_seconds:
                return index
            if segment.start_seconds > second:
                return index
        return len(segments) - 1 if segments else None

    @staticmethod
    def _find_block_index(blocks: list[tuple[int, int]], segment_index: int) -> int:
        for block_index, (start_index, end_index) in enumerate(blocks):
            if start_index <= segment_index <= end_index:
                return block_index
        return len(blocks) - 1

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        return {
            token.lower()
            for token in TOKEN_PATTERN.findall(text)
            if token.lower() not in STOPWORDS and len(token) > 2
        }

    @staticmethod
    def _is_reaction_line(text: str) -> bool:
        lowered = text.lower()
        return any(phrase in lowered for phrase in REACTION_PHRASES)

    @staticmethod
    def _has_parallel_quote_pattern(text: str) -> bool:
        lowered = text.lower()
        return any(pattern.search(lowered) for pattern in PARALLEL_QUOTE_PATTERNS)

    @staticmethod
    def _is_motivational_text(text: str) -> bool:
        lowered = text.lower()
        return any(pattern.search(lowered) for pattern in MOTIVATIONAL_PATTERNS)

    @staticmethod
    def _carries_payoff(text: str) -> bool:
        lowered = text.lower()
        return any(pattern.search(lowered) for pattern in PAYOFF_PATTERNS)
