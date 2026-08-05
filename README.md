# Babel Fish

Speech to text that runs entirely in your browser. The audio is processed on your own machine using OpenAI Whisper
via Transformers.js.

Try it here: https://marklamer.github.io/BabelFish/

## How to use it

1. Open the page and give it a moment. The badge in the corner says
   "preparing model" while it downloads the speech model, and "model ready"
   when it is good to go. This download only happens once. After that the
   browser keeps it cached.
2. Drop an audio file onto the big dashed box, or click it to browse.
   Most formats work: m4a, mp3, wav, ogg, flac, mp4.
3. Watch the transcript stream in as it works.
4. Copy the text or download it as a .txt file when it is done.

## Options

- Model: Tiny is fastest, Small is most accurate, Base is the middle ground.
  Bigger models mean a bigger one-time download.
- Language: leave it on Auto-detect, or pick the language of the recording
  for better results, especially on short clips.
- Include timestamps: adds a time range in front of each line. You can
  toggle it after the fact, no need to re-run anything.

The app remembers your choices for next time.

## Good to know

- First visit needs internet for the model download. After that,
  transcription works offline.
- If the badge says GPU (WebGPU) things will be fast. If it says CPU (WASM)
  it still works, just slower.
- Long recordings take a while. The progress bar shows where it is at.
- On a phone the app starts with the Tiny model, since bigger models can
  run out of memory in mobile browsers. If a model still crashes the page,
  the next visit automatically retries with the lightest settings.

## If the site does not work for you

There is a command line fallback in transcribe.py that does the same job
with Python instead of the browser. It is also the better tool for very
long recordings. Open the file - the comments at the top explain what to
install and how to run it.

## Running it locally

No build step. Clone the repo and serve the folder with any static file
server, for example:

    python -m http.server 8765

Then open http://localhost:8765 in your browser.
