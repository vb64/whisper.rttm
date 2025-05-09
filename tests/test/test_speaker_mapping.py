"""Module speaker_mapping.py tests.

make test T=test_speaker_mapping.py
"""
from . import TestBase


class TestSpeakerMapping(TestBase):
    """Module speaker_mapping."""

    def test_get_word_ts_anchor(self):
        """Check get_word_ts_anchor function."""
        from whisper_rttm.speaker_mapping import get_word_ts_anchor

        assert get_word_ts_anchor(2, 6, option='mid') == 4

    def test_speaker_timestamps(self):
        """Check speaker_timestamps function."""
        from whisper_rttm.speaker_mapping import speaker_timestamps

        assert len(speaker_timestamps(self.fixture("short.rttm"))) == 5
        assert len(speaker_timestamps(self.fixture("lukewarm.rttm"))) == 283
