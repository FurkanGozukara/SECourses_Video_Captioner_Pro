# QA verification log - v1.8.0 (2026-09-06)

## Scope

The shared media previews accepted their own uploads and edits while Caption
resolved its input from Files or File path. Replacing an image in the preview
could therefore display one image and caption another. All three previews are
now output-only, and their change callbacks no longer write input state.
Cleared source fields also override pending preview state at Start.

The Caption workspace places the source on the left and caption, Start, Cancel,
and Copy on the right. Additional result formats and export tools are collapsed
under Additional results. Model, Task & Prompt, and Generation are collapsed
below the workspace. Numeric Trim range remains available for video/audio.

## Chrome verification

Tested the production app at `http://127.0.0.1:7870` in installed Google Chrome,
with Gradio 6.26.0, Python 3.12.10, Windows, and an RTX 5090. Real caption runs
used the locally installed Qwen3-Omni Instruct INT4 model.

| Check | Result |
| --- | --- |
| Upload an image with a Unicode filename | `red square ü.png` displayed a red square; the generated caption described a red square, with the same source path in metadata (`0143_qwen3`, 221 tokens, EOS). |
| Try replacing the image through the preview | No clear/upload/recording controls or file input in the preview. A file drop onto the preview was rejected and the selected image remained unchanged. Fullscreen and download controls remain. |
| Clear Files, then Start | Preview disappeared and Start reported Input required; the previous file was not submitted. |
| Upload a replacement and caption it | `blue circle.png` displayed a blue circle and produced a blue-circle caption (`0144_qwen3`, 187 tokens, EOS), with its own source path in metadata. |
| File path, video playback, and numeric trim | `video20s.mp4` played in the read-only video preview. Start=2, End=4 produced a successful caption run whose segment metadata records exactly 2.0–4.0 seconds (`0145_qwen3`). This smoke run deliberately used a 64-token cap. |
| Audio preview | `jfk.wav` displayed waveform, playback, volume, speed, and download controls, without upload, clear, recording, or editing controls. |
| Switch between source modes | File path displayed the selected video/audio; Folder batch with `red*` displayed the red-square source; returning to Upload files restored the selected blue circle. |
| Desktop layout | At 1440×1000 and 1366×768, caption and primary actions are to the right of the media; settings are below both columns. At 1366×768, the focused image workspace fits its preview, source control, caption, Start/Cancel/Copy, and Trim range into one view. |
| Narrow layout | At 600×900, media and results stack vertically, controls remain usable, and there is no horizontal page overflow. |
| Secondary results and themes | Additional results expands with the existing JSON/subtitle/file/clip tabs and export actions; Files displays the generated outputs. Light and dark layouts remain legible. |

Screenshots and browser diagnostics are retained locally in `output/playwright/`.

## Automated verification

- `pytest tests/test_caption_workspace.py tests/test_whisper_ui.py -q`: **25 passed**.
- `pytest tests -q`: **581 passed, 9 skipped, 9 failed**. Every failure below was
  reproduced on the unchanged starting commit, `27ff2d6`, in a detached checkout.
  None is introduced by this release. The existing suite is not fully green.
- Python compilation and `git diff --check` passed.

The new regression tests cover output-only preview wiring in Caption and
Transcribe, cleared and replaced upload/path inputs with stale preview state,
cleared folder inputs, and the workspace's component hierarchy. Existing Whisper
handler tests now supply the authoritative source field, as the browser does.

### Existing suite failures reproduced before this change

| Test | Existing mismatch |
| --- | --- |
| `test_folder_light_scan_reports_caption_coverage` | Expects unescaped `<stem>` in HTML output. |
| `test_zip_upload_descends_wrapper_folder_and_can_enable_recursive_scan` | Expects extraction status at an older output tuple index. |
| `test_non_playable_video_uses_a_poster_without_assigning_gradio_video` | Expects the previous poster help text. |
| `test_model_change_returns_immediately_and_one_run_is_submitted` | Calls the caption handler without its prompt-provenance argument. |
| `test_caption_and_retry_generators_match_wired_outputs` | Calls the caption handler without its prompt-provenance argument. |
| `test_d1_d21_cancel_note_is_separate_and_escape_rearms_after_expiry` | Calls the caption handler without its prompt-provenance argument. |
| `test_pipeline_client_release_model_through_the_worker_protocol` | Expects the worker to remain alive after release. |
| `test_d13_idle_release_and_worker_stop_are_logged` | Its idle-worker fixture does not produce the expected log line. |
| `test_at_most_one_deferred_listener_per_event` | Caption length already has two deferred change listeners. |
