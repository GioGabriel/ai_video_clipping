from src.core.models import SentenceBlock
from src.modules.viral_phrase_classifier.service import ViralPhraseClassifier


def test_classifier_scores_multiple_trigger_groups() -> None:
    classifier = ViralPhraseClassifier()

    results = classifier.classify(
        [
            SentenceBlock(
                start_seconds=0.0,
                end_seconds=4.0,
                text="Most people don't know this, and here's what to do.",
            )
        ]
    )

    assert results[0].phrase_score == 35
    assert results[0].detected_triggers == ["curiosity", "advice"]


def test_classifier_detects_contrarian_emotional_and_list_phrases() -> None:
    classifier = ViralPhraseClassifier()

    results = classifier.classify(
        [
            SentenceBlock(
                start_seconds=10.0,
                end_seconds=14.0,
                text="Three reasons why everything you've been told is wrong and this mistake cost me everything.",
            )
        ]
    )

    assert results[0].phrase_score == 45
    assert results[0].detected_triggers == ["contrarian", "emotional", "list"]
