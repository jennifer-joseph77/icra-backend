import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import config
import database
import rag_pipeline

# These tests exercise _call_anthropic_with_tools / _call_gemini_with_tools directly
# by faking the SDK client objects, rather than going through the `client` fixture
# (whose mock_call_llm fixture stubs out _call_llm_with_tools entirely).


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """An isolated SQLite DB, without the TestClient/Chroma/LLM-mocking machinery."""
    monkeypatch.setattr(config, "SQLITE_DB_PATH", str(tmp_path / "test.db"))
    database.init_db()


def _create_session(session_id: str):
    """knowledge_gaps.session_id has a FK to chat_sessions, so tests that pass a
    session_id must first create a real row for it to reference."""
    conn = database.get_connection()
    conn.execute("INSERT INTO chat_sessions (id, title) VALUES (?, ?)", (session_id, "Test session"))
    conn.commit()
    conn.close()
    return session_id


def _tool_use_block(question, context, tool_use_id="toolu_1"):
    return SimpleNamespace(
        type="tool_use",
        id=tool_use_id,
        name="flag_knowledge_gap",
        input={"question": question, "context": context},
    )


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


# ── Anthropic branch ─────────────────────────────────────────────────────────


def test_anthropic_tool_call_then_final_answer(isolated_db, monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")
    _create_session("sess-1")

    first_response = SimpleNamespace(
        stop_reason="tool_use",
        content=[_tool_use_block("What is the parking fee?", "no docs matched", "toolu_1")],
    )
    second_response = SimpleNamespace(
        stop_reason="end_turn",
        content=[_text_block("I couldn't find that, please contact Student Services.")],
    )
    mock_create = MagicMock(side_effect=[first_response, second_response])
    mock_client = MagicMock()
    mock_client.messages.create = mock_create
    monkeypatch.setattr(rag_pipeline.anthropic, "Anthropic", MagicMock(return_value=mock_client))

    result = rag_pipeline._call_anthropic_with_tools("Student question: what is the parking fee?", session_id="sess-1")

    assert result == "I couldn't find that, please contact Student Services."
    assert mock_create.call_count == 2

    gaps, total = database.get_knowledge_gaps()
    assert total == 1
    assert gaps[0]["question"] == "What is the parking fee?"
    assert gaps[0]["context"] == "no docs matched"
    assert gaps[0]["session_id"] == "sess-1"

    second_call_messages = mock_create.call_args_list[1].kwargs["messages"]
    tool_result_message = second_call_messages[-1]
    assert tool_result_message["role"] == "user"
    assert tool_result_message["content"][0]["tool_use_id"] == "toolu_1"


def test_anthropic_caps_at_max_tool_rounds(isolated_db, monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")

    first_response = SimpleNamespace(
        stop_reason="tool_use",
        content=[_tool_use_block("Question A", "context A", "toolu_1")],
    )
    second_response = SimpleNamespace(
        stop_reason="tool_use",
        content=[_tool_use_block("Question B", "context B", "toolu_2")],
    )
    mock_create = MagicMock(side_effect=[first_response, second_response])
    mock_client = MagicMock()
    mock_client.messages.create = mock_create
    monkeypatch.setattr(rag_pipeline.anthropic, "Anthropic", MagicMock(return_value=mock_client))

    result = rag_pipeline._call_anthropic_with_tools("some question", session_id=None)

    # Loop stops after MAX_TOOL_ROUNDS=1 round-trip, even though the second
    # response also requested a tool call.
    assert mock_create.call_count == 2
    assert result == rag_pipeline._INCOMPLETE_RESPONSE_MESSAGE

    # Only the first tool call gets executed/logged.
    _, total = database.get_knowledge_gaps()
    assert total == 1


def test_anthropic_no_tool_call_single_round_trip(isolated_db, monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")

    response = SimpleNamespace(stop_reason="end_turn", content=[_text_block("Here is your answer.")])
    mock_create = MagicMock(return_value=response)
    mock_client = MagicMock()
    mock_client.messages.create = mock_create
    monkeypatch.setattr(rag_pipeline.anthropic, "Anthropic", MagicMock(return_value=mock_client))

    result = rag_pipeline._call_anthropic_with_tools("some question", session_id=None)

    assert result == "Here is your answer."
    assert mock_create.call_count == 1
    _, total = database.get_knowledge_gaps()
    assert total == 0


# ── Gemini (OpenAI SDK) branch ───────────────────────────────────────────────


def _fake_tool_call(question, context, call_id="call_1", thought_signature="sig-123"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name="flag_knowledge_gap",
            arguments=json.dumps({"question": question, "context": context}),
        ),
        extra_content={"google": {"thought_signature": thought_signature}} if thought_signature else None,
    )


def _fake_completion(tool_calls, content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=tool_calls, content=content))])


def test_gemini_tool_call_then_final_answer(isolated_db, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    _create_session("sess-2")

    first_response = _fake_completion(
        [_fake_tool_call("What is the shuttle schedule?", "no near matches", "call_1")], None
    )
    second_response = _fake_completion(None, "I couldn't find that, please contact Student Services.")
    mock_create = MagicMock(side_effect=[first_response, second_response])
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create
    monkeypatch.setattr(rag_pipeline.openai, "OpenAI", MagicMock(return_value=mock_client))

    result = rag_pipeline._call_gemini_with_tools("Student question: shuttle schedule?", session_id="sess-2")

    assert result == "I couldn't find that, please contact Student Services."
    assert mock_create.call_count == 2

    gaps, total = database.get_knowledge_gaps()
    assert total == 1
    assert gaps[0]["question"] == "What is the shuttle schedule?"
    assert gaps[0]["session_id"] == "sess-2"

    second_call_messages = mock_create.call_args_list[1].kwargs["messages"]
    tool_message = second_call_messages[-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"

    # The thought_signature Gemini attached to the tool call must be echoed
    # back verbatim on the assistant message, or Gemini 3 rejects the request.
    assistant_message = second_call_messages[-2]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "sig-123"}}


def test_gemini_caps_at_max_tool_rounds(isolated_db, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")

    first_response = _fake_completion([_fake_tool_call("Question A", "context A", "call_1")], None)
    second_response = _fake_completion([_fake_tool_call("Question B", "context B", "call_2")], None)
    mock_create = MagicMock(side_effect=[first_response, second_response])
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create
    monkeypatch.setattr(rag_pipeline.openai, "OpenAI", MagicMock(return_value=mock_client))

    result = rag_pipeline._call_gemini_with_tools("some question", session_id=None)

    assert mock_create.call_count == 2
    assert result == rag_pipeline._INCOMPLETE_RESPONSE_MESSAGE
    _, total = database.get_knowledge_gaps()
    assert total == 1


def test_gemini_no_tool_call_single_round_trip(isolated_db, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")

    response = _fake_completion(None, "Here is your answer.")
    mock_create = MagicMock(return_value=response)
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create
    monkeypatch.setattr(rag_pipeline.openai, "OpenAI", MagicMock(return_value=mock_client))

    result = rag_pipeline._call_gemini_with_tools("some question", session_id=None)

    assert result == "Here is your answer."
    assert mock_create.call_count == 1
    _, total = database.get_knowledge_gaps()
    assert total == 0


def test_gemini_tool_call_without_thought_signature_still_serializes(isolated_db, monkeypatch):
    """Non-thinking OpenAI-compatible providers won't set extra_content at all."""
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")

    first_response = _fake_completion(
        [_fake_tool_call("Question A", "context A", "call_1", thought_signature=None)], None
    )
    second_response = _fake_completion(None, "Final answer.")
    mock_create = MagicMock(side_effect=[first_response, second_response])
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create
    monkeypatch.setattr(rag_pipeline.openai, "OpenAI", MagicMock(return_value=mock_client))

    result = rag_pipeline._call_gemini_with_tools("some question", session_id=None)

    assert result == "Final answer."
    assistant_message = mock_create.call_args_list[1].kwargs["messages"][-2]
    assert "extra_content" not in assistant_message["tool_calls"][0]
