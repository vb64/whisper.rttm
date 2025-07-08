"""Transcibe mp3 to json with word data."""
import time
import argparse
import sys

import faster_whisper
from whisper_rttm import Model, Device, MTYPES

VERSION = '1.0'
COPYRIGHTS = 'Copyrights by Vitaly Bogomolov 2025'
PARSER = argparse.ArgumentParser(description='Whisper words transcribe tool.')

PARSER.add_argument(
  "mp3_file",
  help="Audio file for transcribe."
)
PARSER.add_argument(
  "out_file",
  help="Json file for output."
)

sys.path.insert(1, '.')


def whisper_to_json(segments, _total_msec):
    """Decode whisper segments to json."""
    data = []
    for segment in segments:
        seg_data = [segment.start * 1000, (segment.end - segment.start) * 1000, segment.text.strip()]
        words = []
        for word in segment.words:
            words.append([word.start * 1000, (word.end - word.start) * 1000, word.word.strip()])
        seg_data.append(words)
        data.append(seg_data)

    return data


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
      word_timestamps=True
    )
    duration = int(info.duration_after_vad * 1000)
    print("duration", duration, "msec")
    data = whisper_to_json(segments, duration)
    print(data)

    print(options.out_file, "{} sec".format(int(time.time() - stime)))
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(PARSER.parse_args()))
