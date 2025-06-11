"""Transcibe mp3 to json with word data."""
import time
import argparse
import sys

import faster_whisper
from whisper_rttm.srt import whisper_to_srt
from whisper_rttm import Model, Device, MTYPES

VERSION = '1.0'
COPYRIGHTS = 'Copyrights by Vitaly Bogomolov 2025'
PARSER = argparse.ArgumentParser(description='Whisper words transcribe tool.')

PARSER.add_argument(
  "mp3_file",
  help="Audio file for transcribe."
)
PARSER.add_argument(
  "srt_file",
  help="Json file for output."
)

sys.path.insert(1, '.')


def map_speakers(_rttm_file, srt_file, segments, info):
    """Combine Whisper segments and Nemo rttm."""
    # word_timestamps=False,
    #  multilingual=False,
    #  max_new_tokens=None,
    #  hotwords=None
    print("# mp3", int(info.duration * 1000), int(info.duration_after_vad * 1000))

    # rttm = NemoRttm.from_file(rttm_file, int(info.duration * 1000))
    # first = rttm.rows[0]
    # last = rttm.rows[-1]
    # print("# rttm", last.start + last.length - first.start)

    first, last = None, None
    for segment in segments:
        last = segment
        if first is None:
            first = segment

    print("# segment", int((last.end - first.start) * 1000))

    return srt_file


def main(options):  # pylint: disable=too-many-locals
    """Entry point."""
    print("Whisper transcribe tool v.{}. {}".format(VERSION, COPYRIGHTS))
    stime = time.time()

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

    whisper_to_srt(options.srt_file, segments, info)

    print(options.srt_file, "{} sec".format(int(time.time() - stime)))
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(PARSER.parse_args()))
