"""Transcibe mp3 to srt with rttm file."""
import argparse
import sys
import time

import faster_whisper

sys.path.insert(1, '.')
VERSION = '1.2'
COPYRIGHTS = 'Copyrights by Vitaly Bogomolov 2025'
PARSER = argparse.ArgumentParser(description='Whisper transcribe tool.')

PARSER.add_argument(
  "mp3_file",
  help="Mp3 file for transcribe."
)
PARSER.add_argument(
  "srt_file",
  help="Srt file for output."
)

PARSER.add_argument(
  "--rttm",
  default='',
  help="Optional Nemo rttm file."
)
PARSER.add_argument(
  "--lang",
  default='ru',
  help="Audio language. Default is 'ru'"
)
PARSER.add_argument(
  "--whisper_batch",
  type=int,
  default=0,
  help="Batch size for whisper batched inference. Default 0 (original whisper longform inference).",
)
PARSER.add_argument(
  "--torch_batch",
  type=int,
  default=0,
  help="Torch batch size. Default 0 (disabled).",
)


def main(options):
    """Entry point."""
    print("Whisper transcribe tool v.{}. {}".format(VERSION, COPYRIGHTS))
    stime = time.time()

    from whisper_rttm import Model, Device, MTYPES
    from whisper_rttm.transcript import transcribe
    from whisper_rttm.srt import whisper_to_srt

    whisper_model = faster_whisper.WhisperModel(
      Model.Large,
      device=Device.Cpu,
      compute_type=MTYPES[Device.Cpu]
    )
    whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
    waveform = faster_whisper.decode_audio(options.mp3_file)
    suppress_tokens = [-1]

    if options.whisper_batch > 0:
        segments, info = whisper_pipeline.transcribe(
          waveform, options.lang, suppress_tokens=suppress_tokens,
          batch_size=options.whisper_batch,
        )
    else:
        segments, info = whisper_model.transcribe(
          waveform, options.lang, suppress_tokens=suppress_tokens,
          vad_filter=True,
        )

    if options.rttm:
        srt_file = transcribe(
          waveform, options.rttm, options.srt_file,
          segments, info, options.torch_batch
        )
    else:
        srt_file = whisper_to_srt(options.srt_file, segments, info)

    print(srt_file, "{} sec".format(int(time.time() - stime)))
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(PARSER.parse_args()))
