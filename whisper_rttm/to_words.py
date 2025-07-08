"""Transcibe mp3 to json with word data."""
import time
import argparse
import sys
import json

import faster_whisper

sys.path.insert(1, '.')
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


def msec(sec):
    """Return int milliseconds for numpy float seconds."""
    return int(round(float(sec * 1000)))


def whisper_to_json(segments, _total_msec):
    """Decode whisper segments to json."""
    data = []
    for segment in segments:
        seg_data = [msec(segment.start), msec(segment.end - segment.start), segment.text.strip()]
        words = []
        for word in segment.words:
            words.append([msec(word.start), msec(word.end - word.start), word.word.strip()])
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
    print(json.dumps(data, indent=4))

    print(options.out_file, "{} sec".format(int(time.time() - stime)))
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(PARSER.parse_args()))
