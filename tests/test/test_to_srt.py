"""Module to_srt.py tests.

make test T=test_to_srt.py
"""
import pytest

from . import TestBase


class TestToSrt(TestBase):
    """Module to_srt.py."""

    @pytest.mark.longrunning
    def test_main(self):
        """Check main function."""
        from whisper_rttm.to_srt import PARSER, main

        options = PARSER.parse_args([
          "--rttm", self.fixture('short.rttm'),
          self.fixture('short.mp3'),
          self.build('rttm.srt'),
        ])
        assert main(options) == 0

        options = PARSER.parse_args([
          "--whisper_batch", "0",
          self.fixture('short.mp3'),
          self.build('no_rttm_no_batch.srt'),
        ])
        assert main(options) == 0

        options = PARSER.parse_args([
          self.fixture('short.mp3'),
          self.build('no_rttm.srt'),
        ])
        assert main(options) == 0
