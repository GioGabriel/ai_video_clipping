from src.core.timecode import seconds_to_srt_timestamp, seconds_to_timecode, timecode_to_seconds


def test_timecode_to_seconds_supports_hh_mm_ss() -> None:
    assert timecode_to_seconds("01:02:03") == 3723


def test_seconds_to_timecode_round_trip() -> None:
    original = 95.125
    converted = seconds_to_timecode(original)
    assert converted == "00:01:35.125"
    assert timecode_to_seconds(converted) == original


def test_seconds_to_srt_timestamp() -> None:
    assert seconds_to_srt_timestamp(12.345) == "00:00:12,345"
