# Model Downloader Contract

The canonical downloader is the distribution-level `Models_Downloader.py`.
The app should invoke it with the same Python interpreter as the app and stream
merged stdout/stderr as UTF-8:

```text
python -u ../Models_Downloader.py --ensure <catalog-key>
```

## Catalog and layout

The catalog keys are:

- `timechat_bf16`, `timechat_int8`, `timechat_int4`
- `avocado_bf16`, `avocado_int8`, `avocado_int4`
- `qwen3_omni_instruct_bf16`, `qwen3_omni_instruct_int8`, `qwen3_omni_instruct_int4`
- `qwen3_omni_thinking_bf16`, `qwen3_omni_thinking_int8`, `qwen3_omni_thinking_int4`
- `qwen3_omni_captioner_bf16`, `qwen3_omni_captioner_int8`, `qwen3_omni_captioner_int4`

By default key `K` installs under
`SECourses_Video_Captioner_Pro/models/K/`. `VCAP_MODELS_DIR` or
`--target-root` replaces the models root, and the key is still appended.

Every required path is discovered dynamically with
`HfApi.list_repo_tree(..., path_in_repo=<model-subfolder>, recursive=True)`.
The complete listing, sizes, pinned commit, and published LFS SHA-256/Git blob
digests are cached in `download_cache/remote_index_<key>.json` for 24 hours.
`--refresh-index` forces a fresh listing.

## Exact ready rule

A model is ready only when all of these are true:

1. The current remote index contains at least one required file.
2. Every file listed under that model's HF subfolder exists at the matching
   relative path below `models/<key>/`.
3. Every local file has exactly the byte size published by Hugging Face.
4. Every file with a published digest has either a still-valid verified-cache
   record for the same digest, size, and modification time, or has just been
   re-hashed successfully. Files without a published digest must pass size
   verification.

Extra local files do not affect readiness. A `.part` file never counts as
ready. `--ensure` hashes same-sized unverified files before skipping them;
`--verify <key>` always re-hashes. The app must use this rule rather than
checking for a single weight filename or directory.

## CLI

```text
Models_Downloader.py                         # interactive numbered menu
Models_Downloader.py --ensure <key>          # repeat --ensure as needed
Models_Downloader.py --all
Models_Downloader.py --list
Models_Downloader.py --verify <key>
Models_Downloader.py --status --json
```

`--ensure` emits machine-readable lines:

```text
VCAP_STATUS <key> downloading <message>
VCAP_STATUS <key> ready <message>
VCAP_STATUS <key> failed <message>
```

Exit code `0` means every request is ready, `1` means failure, and `2`
means cancellation. Cancellation and network failures retain exact
`.part`/`.part.json` range state for the next invocation.

`--repo`, `--subfolder`, repeatable `--only <glob>`, and
`--max-bytes-for-test` exist for downloader testing. `--only` deliberately
narrows the required set for that test invocation and must not be used by the
app. `--gguf-index` processes recognized
`Video_Captioner_Pro/gguf/` entries if a root `gguf_catalog.json` is
published; absence is a successful no-op.

