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
- `qwen3_omni_instruct_gguf_q4`, `qwen3_omni_instruct_gguf_q8`
- `qwen3_omni_thinking_gguf_q4`, `qwen3_omni_thinking_gguf_q8`
- `qwen3_omni_captioner_gguf_q4`, `qwen3_omni_captioner_gguf_q8`

By default key `K` installs under
`SECourses_Video_Captioner_Pro/models/K/`. `VCAP_MODELS_DIR` or
`--target-root` replaces the models root, and the key is still appended.

Every required path is discovered dynamically with
`HfApi.list_repo_tree(..., path_in_repo=<model-subfolder>, recursive=True)`.
The complete listing, sizes, pinned commit, and published LFS SHA-256/Git blob
digests are cached in `download_cache/remote_index_<key>.json` for 24 hours.
`--refresh-index` forces a fresh listing.

The six GGUF entries are resolved from the non-gated third-party repositories
pinned in `vcap.models.registry` and are downloaded through
`vcap.models.llamacpp_backend.ensure_gguf`. They never use the
`MonsterMMORPG/Wan_GGUF` model tree.

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

GGUF readiness uses the registry's exact model/mmproj filenames and byte
sizes. GGUF verification hashes both files against the pinned SHA-256 values.

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
VCAP_STATUS {"key":"timechat_int4","state":"downloading","fraction":0.423,"bytes_done":2735890432,"bytes_total":6467930328,"message":"Downloading model.safetensors"}
```

The UTF-8 JSON object is emitted for state changes and at most every two
seconds during active transfers. `state` is `downloading`, `verifying`,
`ready`, `error`, `skipped`, or `missing`; byte counts and `fraction` may be
`null`. The app bridge also accepts the older text protocol and plain percent
progress lines.

Exit code `0` means every request is ready, `1` means failure, and `2`
means cancellation. Cancellation and network failures retain exact
`.part`/`.part.json` range state for the next invocation.

`--repo`, `--subfolder`, repeatable `--only <glob>`, and
`--max-bytes-for-test` exist for downloader testing. `--only` deliberately
narrows the required set for that test invocation and must not be used by the
app.
