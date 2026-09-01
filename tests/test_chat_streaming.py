"""Live chat streaming: reasoning-before-answer classification and thought rendering."""

from __future__ import annotations

import pytest

from vcap.models.omni_common import split_thinking
from vcap.pipeline.chat import ChatRequest, _partial_thinking, _StreamAccumulator, run_chat
from vcap.ui.tabs.chat_tab import _chatbot_messages, _is_thought, _last_answer, _thought_message


def test_partial_thinking_hides_open_tag_and_splits_answer() -> None:
    assert _partial_thinking("<thi") == ("", "")
    assert _partial_thinking("<think>\nplan ") == ("plan", "")
    assert _partial_thinking("<think>\nplan more</think>\n\nAnswer") == ("plan more", "\n\nAnswer")
    assert _partial_thinking("Plain answer") == ("", "Plain answer")


def test_split_thinking_treats_unterminated_block_as_reasoning() -> None:
    assert split_thinking("<think>\nstill thinking") == ("still thinking", "")
    assert split_thinking("<think>\nwhy</think>\n\nBecause.") == ("why", "Because.")
    assert split_thinking("No tags at all") == ("", "No tags at all")


def test_stream_accumulator_reports_reasoning_then_answer_deltas() -> None:
    events: list[dict] = []
    accumulator = _StreamAccumulator(events.append, thinking=True)
    for chunk in ("<thi", "nk>\nlook ", "closely</think>\n\nThe ", "door opens."):
        accumulator(chunk, {"delta": chunk, "reasoning_delta": ""})
    assert [event["text"] for event in events] == ["", "", "\n\nThe ", "\n\nThe door opens."]
    assert [event["reasoning"] for event in events] == ["", "look", "look closely", "look closely"]
    assert "".join(event["reasoning_delta"] for event in events) == "look closely"
    assert "".join(event["delta"] for event in events) == "\n\nThe door opens."

    direct: list[dict] = []
    accumulator = _StreamAccumulator(direct.append, thinking=True)
    accumulator("", {"delta": "", "reasoning_delta": "server-side "})
    accumulator("", {"delta": "", "reasoning_delta": "reasoning"})
    accumulator("Answer", {"delta": "Answer", "reasoning_delta": ""})
    assert direct[-1]["reasoning"] == "server-side reasoning"
    assert direct[-1]["text"] == "Answer"


@pytest.mark.parametrize(
    ("model_key", "enable_thinking", "expect_reasoning"),
    [
        ("qwen3_omni_thinking_int4", True, True),
        ("qwen3_omni_thinking_int4", False, False),
        ("qwen3_omni_instruct_int4", True, False),
    ],
)
def test_mock_chat_streams_reasoning_before_answer(
    monkeypatch: pytest.MonkeyPatch,
    model_key: str,
    enable_thinking: bool,
    expect_reasoning: bool,
) -> None:
    monkeypatch.setenv("VCAP_FAKE_CHAT", "1")
    monkeypatch.delenv("VCAP_FAKE_CHAT_DELAY", raising=False)
    events: list[dict] = []
    response = run_chat(
        ChatRequest.from_dict(
            {
                "settings": {"model_key": model_key, "subprocess_mode": False},
                "history": [{"role": "user", "content": "Why is the sky blue?"}],
                "generation": {"enable_thinking": enable_thinking, "max_new_tokens": 64},
            }
        ),
        events.append,
    )
    deltas = [event for event in events if event.get("ev") == "delta"]
    reasoning_deltas = [event for event in deltas if event["reasoning_delta"]]
    answer_deltas = [event for event in deltas if event["delta"]]
    assert answer_deltas and response.text == "Mock answer to: Why is the sky blue?"
    if expect_reasoning:
        assert reasoning_deltas and all(not event["text"] for event in reasoning_deltas)
        assert deltas.index(reasoning_deltas[-1]) < deltas.index(answer_deltas[0])
        assert response.reasoning == "Mock reasoning about: Why is the sky blue?"
        assert response.raw_text.startswith("<think>") and "</think>" in response.raw_text
    else:
        assert not reasoning_deltas and response.reasoning == "" and response.raw_text == response.text


def test_thought_messages_render_and_are_skipped_by_copy_helpers() -> None:
    pending = _thought_message("half a plan", done=False)
    assert pending["metadata"] == {"title": "🧠 Thinking…", "status": "pending"}
    finished = _thought_message("the plan", done=True, seconds=12.345)
    assert finished["metadata"] == {"title": "🧠 Reasoning", "status": "done", "duration": 12.3}
    assert _is_thought(pending) and not _is_thought({"role": "assistant", "content": "answer"})

    shown = _chatbot_messages(
        [
            {"role": "user", "content": "look", "media": ["C:/clips/a.mp4"]},
            {"role": "assistant", "content": "ok", "reasoning": "  why  ", "reasoning_s": 2.0},
            {"role": "assistant", "content": "plain"},
        ]
    )
    assert shown == [
        {"role": "user", "content": "look\n\n📎 a.mp4"},
        {"role": "assistant", "content": "why", "metadata": {"title": "🧠 Reasoning", "status": "done", "duration": 2.0}},
        {"role": "assistant", "content": "ok"},
        {"role": "assistant", "content": "plain"},
    ]
    assert _last_answer(shown) == "plain"
    assert _last_answer(shown[:2]) == ""
