"""Transcibe mp3 to srt with rttm file."""
import argparse
import sys
import time

import faster_whisper
from ctc_forced_aligner import load_alignment_model

sys.path.insert(1, '.')
VERSION = '1.0'
COPYRIGHTS = 'Copyrights by Vitaly Bogomolov 2025'
PARSER = argparse.ArgumentParser(description='Whisper transcribe tool.')

PARSER.add_argument(
  "mp3_file",
  help="Mp3 file for transcribe."
)
PARSER.add_argument(
  "rttm_file",
  help="Input Nemo rttm file."
)
PARSER.add_argument(
  "srt_file",
  help="Srt file for output."
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

    from whisper_rttm import Model, Device, MTYPES, TTYPES
    from whisper_rttm.transcript import transcribe

    alignment_model, alignment_tokenizer = load_alignment_model(
      Device.Cpu,
      dtype=TTYPES[Device.Cpu]
    )
    whisper_model = faster_whisper.WhisperModel(
      Model.Large,
      device=Device.Cpu,
      compute_type=MTYPES[Device.Cpu]
    )
    whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)

    srt_file = transcribe(
      options.mp3_file, options.rttm_file, options.srt_file,
      whisper_pipeline,
      whisper_model,
      alignment_model,
      alignment_tokenizer,
      [-1],
      options.whisper_batch,
      options.torch_batch,
      'ru'
    )

    print(srt_file, "{} sec".format(int(time.time() - stime)))


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(PARSER.parse_args()))
