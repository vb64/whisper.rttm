"""Voice transcription stuff."""
import torch
import faster_whisper
from ctc_forced_aligner import (
  generate_emissions,
  get_alignments,
  get_spans,
  postprocess_results,
  preprocess_text,
)

from .language import LANGS_TO_ISO
from .speaker_mapping import map_speakers
from .srt import write_srt


def transcribe(  # pylint: disable=too-many-locals,too-many-positional-arguments
  mp3_file, rttm_file, srt_file,
  whisper_pipeline,
  whisper_model,
  alignment_model,
  alignment_tokenizer,
  suppress_tokens,
  batch_size,  # whisper batch_size
  torch_batch,
  lang,
):
    """Transcribe the audio file."""
    waveform = faster_whisper.decode_audio(mp3_file)

    if batch_size > 0:
        segments, info = whisper_pipeline.transcribe(
          waveform, lang, suppress_tokens=suppress_tokens,
          batch_size=batch_size,
        )
    else:
        segments, info = whisper_model.transcribe(
          waveform, lang, suppress_tokens=suppress_tokens,
          vad_filter=True,
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
