"""Module to_srt.py tests.

make test T=test_to_srt.py
"""
import pytest
import faster_whisper

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
          self.fixture('short.mp3'),
          self.build('no_rttm.srt'),
        ])
        assert main(options) == 0

    def test_map_speakers(self):
        """Check map_speakers function."""
        from whisper_rttm.to_srt import map_speakers
        from whisper_rttm import Model, Device, MTYPES

        whisper_model = faster_whisper.WhisperModel(
          Model.Large,
          device=Device.Cpu,
          compute_type=MTYPES[Device.Cpu]
        )
        segments, info = whisper_model.transcribe(
          faster_whisper.decode_audio(self.fixture('short.mp3')),
          'ru',
          suppress_tokens=[-1],
          vad_filter=True,
        )

        assert map_speakers(self.fixture('short.rttm'), self.build('short.srt'), segments, info)
