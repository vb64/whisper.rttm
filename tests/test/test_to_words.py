"""Module to_words.py tests.

make test T=test_to_words.py
"""
import pytest
import faster_whisper

from . import TestBase


class TestToWords(TestBase):
    """Module to_words.py."""

    @pytest.mark.longrunning
    def test_main(self):
        """Check main function."""
        from whisper_rttm.to_words import PARSER, main

        options = PARSER.parse_args([
          self.fixture('short.mp3'),
          self.build('no_rttm.srt'),
        ])
        assert main(options) == 0

    def test_whisper_to_json(self):
        """Check map_speakers function."""
        from whisper_rttm.to_words import whisper_to_json
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
          word_timestamps=True
        )
        #  multilingual=False,
        #  max_new_tokens=None,
        #  hotwords=None
        assert int(info.duration * 1000) == 19592
        assert int(info.duration_after_vad * 1000) == 11736
        # rttm = NemoRttm.from_file(rttm_file, int(info.duration * 1000))
        # first = rttm.rows[0]
        # last = rttm.rows[-1]
        # print("# rttm", last.start + last.length - first.start)

        data = whisper_to_json(segments, int(info.duration_after_vad * 1000))
        assert len(data) == 5
