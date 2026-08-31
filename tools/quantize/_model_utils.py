"""Model, prompt, and PyAV media helpers shared by verification tools."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


TIMECHAT_PROMPT = (
    "Thoroughly describe everything in the video, capturing every detail. Include as much "
    "information from the audio as possible, and ensure that the descriptions of both audio "
    "and video are well-coordinated."
)
AVOCADO_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
    "perceiving auditory and visual inputs, as well as generating text and speech."
)
AVOCADO_PROMPT = (
    "Provide a comprehensive description of all the content in the video, leaving out no details. "
    "Be sure to include as much of the audio information as possible, and ensure that your "
    "descriptions of the audio and video are closely aligned."
)


@dataclass(frozen=True)
class ModelIdentity:
    family: str
    key: str
    model_class: type
    config_class: type
    processor_class: type


@dataclass(frozen=True)
class GenerationRecipe:
    modality: str
    max_pixels: int | None
    max_new_tokens: int
    do_sample: bool
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None


def identify_model(model_dir: str | os.PathLike[str]) -> ModelIdentity:
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    key = model_dir.name.lower()
    if config.get("model_type") == "qwen2_5_omni":
        from transformers import (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerConfig,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        return ModelIdentity(
            "qwen2_5_omni",
            key,
            Qwen2_5OmniThinkerForConditionalGeneration,
            Qwen2_5OmniThinkerConfig,
            Qwen2_5OmniProcessor,
        )
    if config.get("model_type") == "qwen3_omni_moe":
        from transformers import (
            Qwen3OmniMoeProcessor,
            Qwen3OmniMoeThinkerConfig,
            Qwen3OmniMoeThinkerForConditionalGeneration,
        )

        return ModelIdentity(
            "qwen3_omni_moe",
            key,
            Qwen3OmniMoeThinkerForConditionalGeneration,
            Qwen3OmniMoeThinkerConfig,
            Qwen3OmniMoeProcessor,
        )
    raise ValueError(f"Unsupported model_type in {model_dir / 'config.json'}")


def generation_recipe(identity: ModelIdentity) -> GenerationRecipe:
    if "captioner" in identity.key and identity.family == "qwen3_omni_moe":
        return GenerationRecipe("audio", None, 2048, True, 0.6, 0.95, 20)
    if "timechat" in identity.key:
        return GenerationRecipe("video", 297_920, 4096, False)
    if "avocado" in identity.key:
        return GenerationRecipe("video", 401_408, 2048, False)
    return GenerationRecipe("video", 401_408, 2048, False)


def instantiate_meta(identity: ModelIdentity, model_dir: str | os.PathLike[str]):
    config = identity.config_class.from_pretrained(model_dir)
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = identity.model_class(config)
    return model


def move_inputs(inputs: Any, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> dict:
    moved = {}
    for name, value in dict(inputs).items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(dtype)
        moved[name] = value
    return moved


def text_prompt_inputs(processor, identity: ModelIdentity) -> tuple[str, dict]:
    prompt = "Describe the role of sound and motion in a short video in one concise sentence."
    conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    chat_text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(text=chat_text, return_tensors="pt", padding=True)
    return chat_text, dict(inputs)


def decode_video_frames(
    path: str | os.PathLike[str], *, fps: float, max_pixels: int, factor: int
) -> list[Image.Image]:
    import av
    from qwen_omni_utils import smart_resize

    frames: list[Image.Image] = []
    next_time = 0.0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        guessed_rate = float(stream.average_rate or stream.guessed_rate or fps)
        for index, frame in enumerate(container.decode(stream)):
            timestamp = float(frame.time) if frame.time is not None else index / guessed_rate
            if timestamp + 1e-6 < next_time:
                continue
            image = frame.to_image().convert("RGB")
            height, width = smart_resize(
                image.height,
                image.width,
                factor=factor,
                min_pixels=factor * factor,
                max_pixels=max_pixels,
            )
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            frames.append(image)
            next_time += 1.0 / fps
    if not frames:
        raise ValueError(f"No video frames decoded from {path}")
    if len(frames) % 2:
        frames.append(frames[-1].copy())
    return frames


def decode_audio(path: str | os.PathLike[str], sample_rate: int = 16_000) -> np.ndarray:
    environment = os.environ.copy()
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True, env=environment)
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not audio.size:
        raise ValueError(f"No audio decoded from {path}")
    return audio


def multimodal_inputs(
    processor,
    identity: ModelIdentity,
    media_path: str | os.PathLike[str],
) -> tuple[dict, GenerationRecipe]:
    recipe = generation_recipe(identity)
    media_path = Path(media_path)
    if recipe.modality == "audio":
        conversation = [{"role": "user", "content": [{"type": "audio"}]}]
        chat_text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audio = decode_audio(media_path)
        values = processor(
            text=chat_text,
            audio=[audio],
            use_audio_in_video=False,
            return_tensors="pt",
            padding=True,
        )
        return dict(values), recipe

    system = None
    prompt = "Describe the video."
    if "timechat" in identity.key:
        prompt = TIMECHAT_PROMPT
    elif "avocado" in identity.key:
        system = AVOCADO_SYSTEM
        prompt = AVOCADO_PROMPT
    conversation = []
    if system:
        conversation.append({"role": "system", "content": [{"type": "text", "text": system}]})
    conversation.append(
        {
            "role": "user",
            "content": [{"type": "video"}, {"type": "text", "text": prompt}],
        }
    )
    chat_text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    factor = 28 if identity.family == "qwen2_5_omni" else 32
    assert recipe.max_pixels is not None
    frames = decode_video_frames(media_path, fps=2.0, max_pixels=recipe.max_pixels, factor=factor)
    audio = decode_audio(media_path)
    values = processor(
        text=chat_text,
        audio=[audio],
        videos=[frames],
        fps=2.0,
        use_audio_in_video=True,
        return_tensors="pt",
        padding=True,
    )
    return dict(values), recipe


@torch.inference_mode()
def last_token_logits(model, inputs: dict) -> torch.Tensor:
    forward_inputs = {
        key: value
        for key, value in inputs.items()
        if key in {"input_ids", "attention_mask", "position_ids"}
    }
    output = model(**forward_inputs, use_cache=False)
    return output.logits[:, -1, :].float().cpu()


@torch.inference_mode()
def generate_with_metrics(model, processor, inputs: dict, recipe: GenerationRecipe) -> dict:
    torch.cuda.reset_peak_memory_stats()
    input_tokens = int(inputs["input_ids"].shape[1])

    # A logits processor runs after every generation forward. CUDA events
    # separate prefill from decode without timing an unrelated cold forward.
    class _GenerationTimer:
        def __init__(self) -> None:
            self.events: list[torch.cuda.Event] = []

        def __call__(self, input_ids, scores):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.events.append(event)
            return scores

    from transformers import LogitsProcessorList

    timer = _GenerationTimer()

    kwargs = {
        "max_new_tokens": recipe.max_new_tokens,
        "do_sample": recipe.do_sample,
    }
    if recipe.modality == "video":
        kwargs["use_audio_in_video"] = True
    if recipe.do_sample:
        kwargs.update(
            temperature=recipe.temperature,
            top_p=recipe.top_p,
            top_k=recipe.top_k,
        )
    torch.cuda.synchronize()
    generation_start = torch.cuda.Event(enable_timing=True)
    generation_start.record()
    started = time.perf_counter()
    output_ids = model.generate(
        **inputs,
        **kwargs,
        logits_processor=LogitsProcessorList([timer]),
    )
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - started
    new_ids = output_ids[:, input_tokens:]
    generated_tokens = int(new_ids.shape[1])
    caption = processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    if timer.events:
        prefill_seconds = generation_start.elapsed_time(timer.events[0]) / 1000.0
    else:
        prefill_seconds = generation_seconds
    decode_intervals = min(generated_tokens, len(timer.events)) - 1
    if decode_intervals > 0:
        decode_seconds = timer.events[0].elapsed_time(timer.events[-1]) / 1000.0
        decode_tok_s = decode_intervals / max(decode_seconds, 1e-9)
    else:
        decode_seconds = 0.0
        decode_tok_s = 0.0
    return {
        "caption": caption,
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "prefill_seconds": prefill_seconds,
        "generation_seconds": generation_seconds,
        "prefill_tok_s": input_tokens / max(prefill_seconds, 1e-9),
        "decode_seconds": decode_seconds,
        "decode_tok_s": decode_tok_s,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
