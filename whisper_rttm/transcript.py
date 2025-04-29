"""Voice transcription stuff."""
import torch
from ctc_forced_aligner import (
  load_alignment_model,
  generate_emissions,
  get_alignments,
  get_spans,
  postprocess_results,
  preprocess_text,
)

from .language import LANGS_TO_ISO
from .speaker_mapping import map_speakers
from .srt import write_srt
from . import Device, TTYPES


def transcribe(  # pylint: disable=too-many-locals
  waveform, rttm_file, srt_file,
  segments, info, torch_batch,
):
    """Transcribe the audio file."""
    alignment_model, alignment_tokenizer = load_alignment_model(
      Device.Cpu,
      dtype=TTYPES[Device.Cpu]
    )
    emissions, stride = generate_emissions(
      alignment_model,
      torch.from_numpy(waveform)
      .to(alignment_model.dtype)
      .to(alignment_model.device),
      batch_size=torch_batch,
    )
    tokens_starred, text_starred = preprocess_text(
      "".join(segment.text for segment in segments),
      romanize=True,
      language=LANGS_TO_ISO[info.language],
    )
    align_segments, scores, blank_token = get_alignments(
      emissions,
      tokens_starred,
      alignment_tokenizer,
    )
    spans = get_spans(tokens_starred, align_segments, blank_token)
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)
    write_srt(
      map_speakers(rttm_file, word_timestamps),
      srt_file
    )

    return srt_file
