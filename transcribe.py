# Command line fallback for the Babel Fish website. Does the same job on your
# own machine with Python, which is handy for very long recordings or if the
# site is not working for you.
#
# What you need installed:
#   1. Python 3.9 or newer: https://www.python.org/downloads/
#   2. The Whisper library:  pip install openai-whisper
#      (this pulls in PyTorch as well, so the download is large)
#   3. ffmpeg, which Whisper uses to read audio files: https://ffmpeg.org/download.html
#      Either add its bin folder to your system PATH yourself, or fill in
#      FFMPEG_BIN below and the script will add it for this run.
#
# Usage:
#   python transcribe.py "path\to\recording.m4a"
#
# The transcript is saved as a .txt file next to the recording.

import sys
import os
import datetime

# [link to your ffmpeg bin folder, e.g. C:\ffmpeg\bin]
# Leave empty if ffmpeg is already on your PATH.
FFMPEG_BIN = r""
if FFMPEG_BIN:
    os.environ["PATH"] += os.pathsep + FFMPEG_BIN
os.environ["PYTHONUTF8"] = "1"  # fix for special characters

import whisper


def now():
    return datetime.datetime.now().strftime("%H:%M:%S")


if len(sys.argv) < 2:
    print('Usage: python transcribe.py "path\\to\\recording.m4a"')
    sys.exit(1)

audio_path = sys.argv[1]
if not os.path.isfile(audio_path):
    print(f"File not found: {audio_path}")
    sys.exit(1)

print(f"[{now()}] Loading model...")

# Options: tiny, base, small, medium, large. Bigger is more accurate but slower.
model = whisper.load_model("base")

print(f"[{now()}] Model loaded. Starting transcription...")
print("(This may take 10-30 minutes for a 30-minute recording)")

result = model.transcribe(audio_path, verbose=True)

print(f"[{now()}] Transcription complete. Saving...")

out_path = os.path.splitext(audio_path)[0] + ".txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(result["text"])

print(f"Bazinga! Transcript saved to {out_path}")
