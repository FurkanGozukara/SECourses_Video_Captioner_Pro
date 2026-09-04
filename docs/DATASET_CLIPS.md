# Dataset clip captions

Dataset clip captions keep the visual description, speech or sound description, and trainer-facing caption as separate, reproducible artifacts. The layout is enabled whenever **Audio caption source** is not `none`.

## Basic layout

For an input named `myVideo.mp4`, one caption unit writes:

```text
<out_dir>/
  myVideo.txt                    # optional merged caption
  myVideo.json                   # optional ordinary structured output
  myVideo.srt                    # optional ordinary subtitle output
  video_caption/
    myVideo.txt                  # clean, fully post-processed video caption
  audio_caption/
    myVideo.txt                  # rendered speech and/or sound caption
```

`<out_dir>` is the numbered single-run directory, a mirrored batch output directory, or the source file's directory when **Save outputs next to the source files** is enabled. `metadata.json`, `run_log.txt`, and `.work/` always remain in the numbered run directory.

Other requested formats retain their normal locations. Structured JSON also includes `video_caption`, `audio_caption`, and `merged_caption`, alongside the existing transcript block. Stage 7 transcript sidecars are independent; an empty transcript-format selection keeps the Whisper result in memory without writing sidecars.

## Examples

Single input:

```text
outputs/0001_<model>/
  clip.txt
  metadata.json
  run_log.txt
  video_caption/clip.txt
  audio_caption/clip.txt
```

Mirrored recursive batch:

```text
outputs/batch_captions/session_a/
  clip.txt
  video_caption/clip.txt
  audio_caption/clip.txt
```

Next-to-source batch:

```text
dataset/session_a/
  clip.mp4
  clip.txt
  video_caption/clip.txt
  audio_caption/clip.txt
outputs/batch_0001_<model>/
  metadata.json
  run_log.txt
```

Scene, fixed, or trainer split with saved clips:

```text
clip_clips/
  clip_0001.mp4
  clip_0001.txt
  video_caption/clip_0001.txt
  audio_caption/clip_0001.txt
```

Temporary multi-segment runs use `<stem>_segments/` instead. The item-level timestamped combined caption receives the same three-file layout.

## Sources and ordering

`whisper` uses the stage 7 Whisper settings. It runs even when the stage 7 checkbox is off, because the audio-caption setting requires its in-memory result. Segment units receive only their overlapping transcript with timestamps shifted to start at zero. Prompt injection remains an independent setting.

`captioner` uses a prompt-free `qwen3_omni_captioner_*` checkpoint. `auto` matches the selected main model's INT4, INT8, BF16, GGUF Q4, or GGUF Q8 backend. TimeChat and AVoCaDO map INT4 to Captioner INT4 and INT8 to Captioner INT8; BF16 maps to Captioner INT8 below the 80 GB tier. Audio is extracted as 16 kHz mono and split into windows no longer than 30 seconds. Silent media and files without an audio track produce an empty part.

`both` renders Whisper speech followed by the Captioner sound description. Jobs run in three phases: video captions, sound captions, then merge. Existing-caption mode skips the video-caption phase and never loads its model.

## Templates

The audio template accepts:

- `{{TRANSCRIPT}}`: rendered Whisper text
- `{{SOUND_CAPTION}}`: raw joined sound description
- `{{FILENAME}}`: caption-unit stem

The merge template additionally accepts:

- `{{VIDEO_CAPTION}}`: clean video-caption part
- `{{AUDIO_CAPTION}}`: fully rendered audio-caption part

Empty tokens and their adjacent blank whitespace collapse cleanly. Output is trimmed while meaningful internal newlines are retained.

## Existing captions

With **Video caption source: Reuse existing**, each input file is one unit and splitting is ignored. The lookup order is:

1. `<out_dir>/video_caption/<stem>.txt`
2. `<out_dir>/<stem>.txt`
3. `<source_dir>/<source_stem>.txt`

A plain sidecar is copied into `video_caption/` before merging. Later runs always rebuild from that clean part, so an overwrite run cannot append the same audio caption twice. If no video caption is found, the audio part is still saved, the merged file is omitted, and the item finishes with a diagnostic message.

## Empty audio and skip rules

The `skip` empty policy omits `audio_caption/<stem>.txt` and makes the merged file exactly the video caption. With overwrite enabled, a stale audio part is removed. The `placeholder` policy writes the configured text, `No speech.` by default.

When overwrite is off, ordinary split runs with merged output skip on an existing merged file. Separate-only generated runs require both video and audio parts. Existing-caption runs require the audio part and, when enabled, the merged file. Retry, item limits, media-kind and name filters, ZIP batches, Unicode paths, and recursive mirroring use the same decisions.

The Caption Editor edits only the main `<stem>.txt`. It shows available video and audio parts in a collapsed read-only panel. Regenerating a selected item with a video part updates that clean part and rebuilds the main caption with its existing audio part and current merge template.
