"""Transcibe mp3 to srt with rttm file."""
import argparse
import sys
import time

import faster_whisper

sys.path.insert(1, '.')
VERSION = '1.1'
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


def map_speakers(_rttm_file, srt_file, _segments, _info):
    """Combine Whisper segments and Nemo rttm."""
    return srt_file


def main(options):  # pylint: disable=too-many-locals
    """Entry point."""
    print("Whisper transcribe tool v.{}. {}".format(VERSION, COPYRIGHTS))
    stime = time.time()

    from whisper_rttm import Model, Device, MTYPES
    from whisper_rttm.srt import whisper_to_srt

    whisper_model = faster_whisper.WhisperModel(
      Model.Large,
      device=Device.Cpu,
      compute_type=MTYPES[Device.Cpu]
    )
    waveform = faster_whisper.decode_audio(options.mp3_file)

    segments, info = whisper_model.transcribe(
      waveform, 'ru', suppress_tokens=[-1],
      vad_filter=True,
    )

    if options.rttm:
        srt_file = map_speakers(options.rttm, options.srt_file, segments, info)
    else:
        srt_file = whisper_to_srt(options.srt_file, segments, info)

    print(srt_file, "{} sec".format(int(time.time() - stime)))
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(PARSER.parse_args()))
