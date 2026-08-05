import sys
import os
sys.path.insert(0, "C:\\ml")
os.environ["PATH"] += os.pathsep + r"C:\Users\Mark.Nistor\ffmpeg\ffmpeg-master-latest-win64-gpl\bin"
os.environ["PYTHONUTF8"] = "1"  # fix for special characters

import whisper
import datetime

print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Loading model...")

model = whisper.load_model("base")

print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Model loaded. Starting transcription...")
print("(This may take 10-30 minutes for a 30-minute recording)")

result = model.transcribe(
    r"C:\Users\dacia\Documents\Sound Recordings\Recording.m4a",
    verbose=True
)

print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Transcription complete. Saving...")

with open(r"C:\Users\dacia\Documents\Sound Recordings\transcript.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

print("Bazinga! Transcript saved.")