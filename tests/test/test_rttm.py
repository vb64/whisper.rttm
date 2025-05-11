"""Module rttm.py tests.

make test T=test_rttm.py
"""
from . import TestBase


class TestRttm(TestBase):
    """Module rttm."""

    def test_format_timestamp(self):
        """Check format_timestamp function."""
        from whisper_rttm.rttm import NemoRttm

        rttm = NemoRttm.from_file(self.fixture('short.rttm'), 1000)
        assert len(rttm.rows) == 4
