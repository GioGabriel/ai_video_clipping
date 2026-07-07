from __future__ import annotations

import re

from src.core.models import SentenceBlock

TRIGGER_PATTERNS: dict[str, tuple[int, tuple[re.Pattern[str], ...]]] = {
    "curiosity": (
        20,
        (
            re.compile(r"\bmost people\b", re.IGNORECASE),
            re.compile(r"\bnobody talks about\b", re.IGNORECASE),
            re.compile(r"\bwhat nobody tells you\b", re.IGNORECASE),
            re.compile(r"\byou won't believe\b", re.IGNORECASE),
            re.compile(r"\bi wish i knew\b", re.IGNORECASE),
            re.compile(r"\bwhy does\b", re.IGNORECASE),
            re.compile(r"\beveryone\b.*\bno one\b", re.IGNORECASE),
            re.compile(r"\beverybody\b.*\bno one\b", re.IGNORECASE),
            re.compile(r"\bpeople\b.*\bbut\b.*\bnot\b", re.IGNORECASE),
        ),
    ),
    "contrarian": (
        20,
        (
            re.compile(r"\beveryone thinks\b", re.IGNORECASE),
            re.compile(r"\bbut actually\b", re.IGNORECASE),
            re.compile(r"\bthis is wrong\b", re.IGNORECASE),
            re.compile(r"\bthe truth is\b", re.IGNORECASE),
            re.compile(r"\bthe truth about\b", re.IGNORECASE),
            re.compile(r"\beverything you've been told\b.*\bwrong\b", re.IGNORECASE),
            re.compile(r"\bactually terrible\b", re.IGNORECASE),
        ),
    ),
    "advice": (
        15,
        (
            re.compile(r"\byou should\b", re.IGNORECASE),
            re.compile(r"\bstop doing this\b", re.IGNORECASE),
            re.compile(r"\bstart doing this\b", re.IGNORECASE),
            re.compile(r"\bhere'?s what to do\b", re.IGNORECASE),
            re.compile(r"\bhere(?:'s| is)\s+what\s+to\s+do\b", re.IGNORECASE),
            re.compile(r"\bhere(?:'s| is)\s+how\b", re.IGNORECASE),
            re.compile(r"\bif you want to\b", re.IGNORECASE),
            re.compile(r"\bif you\b.*\byou will\b", re.IGNORECASE),
            re.compile(r"\bthe one thing you need to know\b", re.IGNORECASE),
            re.compile(r"\btrain yourself\b", re.IGNORECASE),
            re.compile(r"\bdo what needs to be done\b", re.IGNORECASE),
            re.compile(r"\bregardless of how you feel\b", re.IGNORECASE),
            re.compile(r"\bfollow through\b", re.IGNORECASE),
            re.compile(r"\bstay consistent\b", re.IGNORECASE),
            re.compile(r"\bconsistent people win\b", re.IGNORECASE),
            re.compile(r"\bdo it scared\b", re.IGNORECASE),
            re.compile(r"\bwhen your why is strong enough\b", re.IGNORECASE),
        ),
    ),
    "emotional": (
        15,
        (
            re.compile(r"\bchanged my life\b", re.IGNORECASE),
            re.compile(r"\bshocked me\b", re.IGNORECASE),
            re.compile(r"\bbiggest mistake\b", re.IGNORECASE),
            re.compile(r"\bdestroyed\b", re.IGNORECASE),
            re.compile(r"\bcost me everything\b", re.IGNORECASE),
            re.compile(r"\bi was shocked\b", re.IGNORECASE),
            re.compile(r"\bjealous\b", re.IGNORECASE),
            re.compile(r"\bmy god\b", re.IGNORECASE),
            re.compile(r"\bregret\b", re.IGNORECASE),
            re.compile(r"\bfear\b", re.IGNORECASE),
            re.compile(r"\bgiving up is the easiest thing\b", re.IGNORECASE),
        ),
    ),
    "list": (
        10,
        (
            re.compile(r"\bone reason\b", re.IGNORECASE),
            re.compile(r"\bthree reasons\b", re.IGNORECASE),
            re.compile(r"\bfive mistakes\b", re.IGNORECASE),
            re.compile(r"\b\d+\s+reasons?\b", re.IGNORECASE),
            re.compile(r"\b\d+\s+mistakes?\b", re.IGNORECASE),
        ),
    ),
    "thesis": (
        20,
        (
            re.compile(r"\bhard now\b.*\beasy later\b", re.IGNORECASE),
            re.compile(r"\beasy now\b.*\bhard later\b", re.IGNORECASE),
            re.compile(r"\bgiving up is the easiest thing\b", re.IGNORECASE),
            re.compile(r"\bno great life has ever been lived by taking the easy route\b", re.IGNORECASE),
            re.compile(r"\bwhen your why is strong enough\b", re.IGNORECASE),
        ),
    ),
}


class ViralPhraseClassifier:
    def classify(self, sentence_blocks: list[SentenceBlock]) -> list[SentenceBlock]:
        return [self._classify_block(block) for block in sentence_blocks]

    def _classify_block(self, block: SentenceBlock) -> SentenceBlock:
        score, detected_triggers = self.classify_text(block.text)

        return SentenceBlock(
            start_seconds=block.start_seconds,
            end_seconds=block.end_seconds,
            text=block.text,
            speaker=block.speaker,
            source_segment_start_index=block.source_segment_start_index,
            source_segment_end_index=block.source_segment_end_index,
            phrase_score=min(score, 100.0),
            detected_triggers=detected_triggers,
        )

    @staticmethod
    def classify_text(text: str) -> tuple[float, list[str]]:
        lowered = text.lower().strip()
        detected_triggers: list[str] = []
        score = 0.0

        for label, (weight, patterns) in TRIGGER_PATTERNS.items():
            if any(pattern.search(lowered) for pattern in patterns):
                detected_triggers.append(label)
                score += weight

        if "?" in lowered and "curiosity" not in detected_triggers:
            detected_triggers.append("curiosity")
            score += 20

        return min(score, 100.0), detected_triggers
