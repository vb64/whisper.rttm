"""Module transcript.py tests.

make test T=test_transcript.py
"""
from pathlib import Path
import pytest

import faster_whisper
# from ctc_forced_aligner import load_alignment_model

from . import TestBase


class TestTranscript(TestBase):
    """Module transcript."""

    @pytest.mark.longrunning
    def test_transcribe(self):
        """Check transcribe function."""
        from whisper_rttm import transcript, Model, Device, MTYPES  # , TTYPES

        # alignment_model, alignment_tokenizer = load_alignment_model(
        #   Device.Cpu,
        #   dtype=TTYPES[Device.Cpu]
        # )
        whisper_model = faster_whisper.WhisperModel(
          Model.Large,
          device=Device.Cpu,
          compute_type=MTYPES[Device.Cpu]
        )
        # whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
        waveform = faster_whisper.decode_audio(self.fixture('short.mp3'))
        segments, info = whisper_model.transcribe(
          waveform, 'ru', suppress_tokens=[-1],
          vad_filter=True,
        )

        srt = transcript.transcribe(
          waveform,
          self.fixture('short.rttm'),
          self.build('short.srt'),
          segments, info, 0
        )
        assert '.srt' in srt
        Path(srt).unlink()
