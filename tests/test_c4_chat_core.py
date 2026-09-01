from __future__ import annotations

import json
from pathlib import Path

import pytest

from vcap.models.base import (
    ChatMessage,
    chat_media_parts,
    chat_media_placeholders,
    normalize_chat_history,
    truncate_chat_history,
)
from vcap.models.llamacpp_backend import build_llamacpp_chat_messages
from vcap.models.omni_common import build_chat_conversation
from vcap.pipeline.chat import ChatRequest, conversation_markdown, save_conversation
from vcap.ui.app import build_app
from vcap.ui.tabs.chat_tab import model_chat_support, resolve_chat_attachments


def test_history_rendering_attaches_media_only_to_first_user_turn() -> None:
    history = [
        ChatMessage("system", "Be concise."),
        ChatMessage("user", "What is shown?"),
        ChatMessage("assistant", "Color bars."),
        ChatMessage("user", "How many?"),
    ]
    media = [{"type": "image"}]
    rendered = build_chat_conversation(history, media)
    assert rendered[0]["role"] == "system"
    assert rendered[1]["content"][0] == {"type": "image"}
    assert rendered[1]["content"][1]["text"] == "What is shown?"
    assert all(
        part.get("type") != "image"
        for message in rendered[2:]
        for part in message["content"]
    )

    openai_messages = build_llamacpp_chat_messages(
        history,
        [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}],
    )
    assert isinstance(openai_messages[1]["content"], list)
    assert openai_messages[-1] == {"role": "user", "content": "How many?"}


def test_history_rendering_attaches_each_turns_own_media() -> None:
    history = [
        ChatMessage("user", "What is shown?", ("a.png",)),
        ChatMessage("assistant", "Color bars."),
        ChatMessage("user", "And in this one?", ("b.mp4",)),
    ]

    def placeholders(index: int, message: ChatMessage) -> list[dict[str, str]]:
        del index
        return [{"type": "image" if path.endswith(".png") else "video"} for path in message.media]

    rendered = build_chat_conversation(history, [{"type": "audio"}], "Be brief.", turn_media=placeholders)
    assert rendered[0]["role"] == "system"
    # Legacy first-turn media leads the first user turn, then that turn's own media.
    assert [part["type"] for part in rendered[1]["content"]] == ["audio", "image", "text"]
    assert [part["type"] for part in rendered[2]["content"]] == ["text"]
    assert [part["type"] for part in rendered[3]["content"]] == ["video", "text"]

    encoded = build_llamacpp_chat_messages(
        history,
        turn_media=lambda index, message: [
            {"type": "image_url", "image_url": {"url": f"data:turn{index}"}}
        ],
    )
    assert encoded[0]["content"][0]["image_url"]["url"] == "data:turn0"
    assert encoded[0]["content"][-1] == {"type": "text", "text": "What is shown?"}
    assert encoded[1] == {"role": "assistant", "content": "Color bars."}
    assert encoded[2]["content"][0]["image_url"]["url"] == "data:turn2"


def test_chat_media_survives_normalization_truncation_and_requests(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
            "0000000c4944415408d763f8ffff3f0005fe02fe0def46b80000000049454e44ae426082"
        )
    )
    history = [
        {"role": "user", "content": "first", "media": [str(image)]},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": "second", "media": str(image)},
    ]
    normalized = normalize_chat_history(history)
    assert normalized[0].media == (str(image),)
    assert normalized[2].media == (str(image),)
    with pytest.raises(ValueError):
        normalize_chat_history([{"role": "assistant", "content": "x", "media": ["y"]}])
    retained, dropped, _ = truncate_chat_history(history, lambda messages: 1, 100)
    assert dropped == 0 and retained[-1].media == (str(image),)
    request = ChatRequest.from_dict(
        {"settings": {"model_key": "qwen3_omni_instruct_int4"}, "history": history}
    )
    assert ChatRequest.from_dict(request.to_dict()) == request
    assert request.history[2].media == (str(image),)
    parts = chat_media_parts([str(image)])
    assert [part.type for part in parts] == ["image"]
    assert chat_media_placeholders(parts) == [{"type": "image"}]
    with pytest.raises(ValueError):
        chat_media_parts([str(tmp_path / "missing.png")])


def test_context_truncation_keeps_media_and_current_turns() -> None:
    history = [
        ChatMessage("system", "S" * 5),
        ChatMessage("user", "media-" + "M" * 10),
        ChatMessage("assistant", "first-" + "A" * 10),
        ChatMessage("user", "middle-one-" + "B" * 40),
        ChatMessage("assistant", "middle-one-answer-" + "C" * 40),
        ChatMessage("user", "middle-two-" + "D" * 40),
        ChatMessage("assistant", "middle-two-answer-" + "E" * 40),
        ChatMessage("user", "current-" + "Q" * 10),
    ]

    def count(messages):
        return sum(len(item.content) for item in messages)

    retained, dropped, tokens = truncate_chat_history(history, count, 100)
    assert dropped == 2
    assert tokens <= 90
    assert [item.role for item in retained] == ["system", "user", "assistant", "user"]
    assert retained[1].content.startswith("media-")
    assert retained[2].content.startswith("first-")
    assert retained[-1].content.startswith("current-")


def test_chat_request_and_conversation_persistence(tmp_path: Path) -> None:
    request = ChatRequest.from_dict(
        {
            "settings": {"model_key": "qwen3_omni_instruct_int4"},
            "history": [{"role": "user", "content": "İstanbul görüntüsü?"}],
            "media": [str(tmp_path / "görüntü.png")],
            "generation": {"max_new_tokens": 64},
            "system_prompt": "Türkçe yanıtla.",
        }
    )
    assert ChatRequest.from_dict(request.to_dict()) == request

    messages = [
        {"role": "user", "content": "What is shown?"},
        {
            "role": "assistant",
            "content": "Six color bars.",
            "reasoning": "I counted each vertical band.",
        },
    ]
    run_dir = save_conversation(
        messages,
        model_key="qwen3_omni_instruct_int4",
        metadata={"system_prompt": "Be exact.", "media": ["görüntü.png"]},
        outputs_root=tmp_path / "outputs",
    )
    assert run_dir.name == "0001_chat_qwen3"
    document = json.loads((run_dir / "conversation.json").read_text(encoding="utf-8"))
    assert document["_meta"]["format"] == "secourses_vcap_conversation"
    assert document["messages"][1]["reasoning"].startswith("I counted")
    markdown = (run_dir / "conversation.md").read_text(encoding="utf-8")
    assert "<summary>Reasoning</summary>" in markdown
    assert "Six color bars." in markdown
    assert conversation_markdown(messages, document["metadata"]).endswith("\n")


def test_chat_support_and_registered_preset_parameters(tmp_path: Path) -> None:
    image = tmp_path / "test image.png"
    image.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
            "0000000c4944415408d763f8ffff3f0005fe02fe0def46b80000000049454e44ae426082"
        )
    )
    assert resolve_chat_attachments([], f'"{image}"') == [str(image.resolve())]
    assert model_chat_support("qwen3_omni_instruct_int4")[0] == "multi"
    assert model_chat_support("timechat_int4")[0] == "single"
    assert model_chat_support("qwen3_omni_captioner_int4")[0] == "unsupported"

    demo = build_app()
    try:
        entries = {entry.key: entry for entry in demo.settings_registry.entries()}
        keys = {
            "chat_temperature",
            "chat_top_p",
            "chat_top_k",
            "chat_max_new_tokens",
            "chat_enable_thinking",
        }
        assert keys <= entries.keys()
        assert all(entries[key].section == "chat" for key in keys)
        assert all(entries[key].in_preset for key in keys)
        assert all(not entries[key].in_metadata for key in keys)
        assert demo.vcap_context.chat_handles is not None
    finally:
        demo.vcap_context.pipeline_client.shutdown()
