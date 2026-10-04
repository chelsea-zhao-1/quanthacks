# extras: spoken research brief

An optional add-on that gives each pipeline run a voice. `audio_brief.py` reads that run's own
results (`data/oldnews/results_<label>/` and `trade_<label>/`), writes a two-minute spoken
summary from a fixed template, and turns it into an MP3 with the ElevenLabs text-to-speech API.

Setup: put `ELEVENLABS_API_KEY=...` in `.env` (never commit it). Then, from the repo root:

    .venv/Scripts/python.exe extras/audio_brief.py --label insample --dry-run   # text only, no API call
    .venv/Scripts/python.exe extras/audio_brief.py --label insample             # -> data/oldnews/audio/brief_insample.mp3

Use `--label holdout` after a sealed-window run. This is not part of the judged notebook.
