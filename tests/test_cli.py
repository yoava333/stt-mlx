import pytest

from stt_mlx.cli import fmt_srt


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (59.9996, "00:01:00,000"),
        (3599.9996, "01:00:00,000"),
    ],
)
def test_fmt_srt_carries_rounded_milliseconds(
    seconds: float, expected: str
) -> None:
    assert fmt_srt(seconds) == expected
