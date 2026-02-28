from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pytest

from afk.llms.clients.shared.content import (
    json_text,
    normalize_role,
    to_input_text_part,
    tool_result_label,
)
from afk.llms.clients.shared.normalization import (
    extract_text_from_content,
    extract_tool_calls,
    extract_usage,
    finalize_stream_tool_calls,
    get_attr,
    get_attr_str,
    to_jsonable,
    to_plain_dict,
)
from afk.llms.clients.shared.transport import collect_headers
from afk.llms.types import ToolCall, Usage


# ---------------------------------------------------------------------------
# content.py -- json_text
# ---------------------------------------------------------------------------


class TestJsonText:
    def test_dict(self):
        result = json_text({"a": 1})
        assert json.loads(result) == {"a": 1}

    def test_list(self):
        result = json_text([1, "two", 3])
        assert json.loads(result) == [1, "two", 3]

    def test_string(self):
        result = json_text("hello")
        assert json.loads(result) == "hello"

    def test_none(self):
        result = json_text(None)
        assert json.loads(result) is None

    def test_non_serializable_uses_str_fallback(self):
        """Non-JSON-serializable objects are coerced via the ``default=str`` fallback."""
        dt = datetime(2025, 1, 15, 12, 0, 0)
        result = json_text(dt)
        parsed = json.loads(result)
        assert isinstance(parsed, str)
        assert "2025" in parsed

    def test_ensure_ascii(self):
        result = json_text({"key": "\u00e9"})
        assert "\\u00e9" in result


# ---------------------------------------------------------------------------
# content.py -- normalize_role
# ---------------------------------------------------------------------------


class TestNormalizeRole:
    @pytest.mark.parametrize("role", ["user", "assistant", "system"])
    def test_supported_roles_returned_as_is(self, role: str):
        assert normalize_role(role) == role

    def test_tool_falls_back_to_user(self):
        assert normalize_role("tool") == "user"

    def test_function_falls_back_to_user(self):
        assert normalize_role("function") == "user"

    def test_empty_string_falls_back_to_user(self):
        assert normalize_role("") == "user"


# ---------------------------------------------------------------------------
# content.py -- tool_result_label
# ---------------------------------------------------------------------------


class TestToolResultLabel:
    def test_with_name(self):
        assert tool_result_label("my_tool") == "my_tool"

    def test_empty_string_returns_tool(self):
        assert tool_result_label("") == "tool"

    def test_none_returns_tool(self):
        assert tool_result_label(None) == "tool"


# ---------------------------------------------------------------------------
# content.py -- to_input_text_part
# ---------------------------------------------------------------------------


class TestToInputTextPart:
    def test_string(self):
        result = to_input_text_part("hello")
        assert result == {"type": "input_text", "text": "hello"}

    def test_int_coerced_to_str(self):
        result = to_input_text_part(42)
        assert result == {"type": "input_text", "text": "42"}

    def test_none_coerced_to_str(self):
        result = to_input_text_part(None)
        assert result == {"type": "input_text", "text": "None"}


# ---------------------------------------------------------------------------
# normalization.py -- to_plain_dict
# ---------------------------------------------------------------------------


class TestToPlainDict:
    def test_dict_returned_as_is(self):
        d = {"a": 1}
        assert to_plain_dict(d) is d

    def test_object_with_model_dump(self):
        class Pydantic:
            def model_dump(self) -> dict:
                return {"x": 10}

        assert to_plain_dict(Pydantic()) == {"x": 10}

    def test_object_with_to_dict(self):
        class SDK:
            def to_dict(self) -> dict:
                return {"y": 20}

        assert to_plain_dict(SDK()) == {"y": 20}

    def test_dataclass(self):
        @dataclass
        class DC:
            name: str = "hi"

        assert to_plain_dict(DC()) == {"name": "hi"}

    def test_object_with_dunder_dict(self):
        class Obj:
            def __init__(self):
                self.z = 30

        assert to_plain_dict(Obj()) == {"z": 30}

    def test_plain_int_returns_empty(self):
        assert to_plain_dict(42) == {}


# ---------------------------------------------------------------------------
# normalization.py -- to_jsonable
# ---------------------------------------------------------------------------


class TestToJsonable:
    def test_none(self):
        assert to_jsonable(None) is None

    def test_str(self):
        assert to_jsonable("abc") == "abc"

    def test_int(self):
        assert to_jsonable(7) == 7

    def test_float(self):
        assert to_jsonable(3.14) == 3.14

    def test_bool(self):
        assert to_jsonable(True) is True

    def test_dict(self):
        assert to_jsonable({"a": 1}) == {"a": 1}

    def test_list(self):
        assert to_jsonable([1, "x"]) == [1, "x"]

    def test_tuple_converted_to_list(self):
        assert to_jsonable((1, 2)) == [1, 2]

    def test_dataclass(self):
        @dataclass
        class DC:
            v: int = 5

        assert to_jsonable(DC()) == {"v": 5}

    def test_non_standard_object_falls_back_to_repr(self):
        """Objects that are not JSON primitives, containers, or convertible via
        ``to_plain_dict`` are converted using ``repr()``."""

        class Weird:
            pass

        result = to_jsonable(Weird())
        assert isinstance(result, str)
        assert "Weird" in result


# ---------------------------------------------------------------------------
# normalization.py -- extract_text_from_content
# ---------------------------------------------------------------------------


class TestExtractTextFromContent:
    def test_string(self):
        assert extract_text_from_content("hello") == "hello"

    def test_list_of_strings(self):
        assert extract_text_from_content(["a", "b"]) == "ab"

    def test_list_of_text_dicts(self):
        items = [{"text": "one"}, {"text": "two"}]
        assert extract_text_from_content(items) == "onetwo"

    def test_list_of_output_text_dicts(self):
        items = [{"type": "output_text", "text": "hi"}]
        assert extract_text_from_content(items) == "hi"

    def test_dict_with_text_key(self):
        assert extract_text_from_content({"text": "val"}) == "val"

    def test_empty_list(self):
        assert extract_text_from_content([]) == ""

    def test_non_dict_items_in_list_skipped(self):
        items = ["a", 123, {"text": "b"}]
        assert extract_text_from_content(items) == "ab"

    def test_int_returns_empty_string(self):
        assert extract_text_from_content(42) == ""


# ---------------------------------------------------------------------------
# normalization.py -- extract_usage
# ---------------------------------------------------------------------------


class TestExtractUsage:
    def test_prompt_and_completion_tokens(self):
        raw = {"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
        usage = extract_usage(raw)
        assert usage == Usage(input_tokens=10, output_tokens=20, total_tokens=30)

    def test_input_and_output_tokens(self):
        raw = {"usage": {"input_tokens": 5, "output_tokens": 15}}
        usage = extract_usage(raw)
        assert usage == Usage(input_tokens=5, output_tokens=15, total_tokens=None)

    def test_prompt_tokens_preferred_over_input_tokens(self):
        raw = {"usage": {"prompt_tokens": 10, "input_tokens": 99, "completion_tokens": 20}}
        usage = extract_usage(raw)
        assert usage.input_tokens == 10

    def test_missing_usage_key(self):
        usage = extract_usage({})
        assert usage == Usage()

    def test_non_dict_usage_returns_default(self):
        usage = extract_usage({"usage": "not_a_dict"})
        assert usage == Usage()

    def test_non_int_tokens_treated_as_none(self):
        raw = {"usage": {"prompt_tokens": "bad", "completion_tokens": 5.5}}
        usage = extract_usage(raw)
        assert usage.input_tokens is None
        assert usage.output_tokens is None


# ---------------------------------------------------------------------------
# normalization.py -- extract_tool_calls
# ---------------------------------------------------------------------------


class TestExtractToolCalls:
    def test_normal_list(self):
        raw = [
            {
                "id": "call_1",
                "function": {"name": "fn", "arguments": {"x": 1}},
            }
        ]
        calls = extract_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0] == ToolCall(id="call_1", tool_name="fn", arguments={"x": 1})

    def test_empty_list(self):
        assert extract_tool_calls([]) == []

    def test_non_list_returns_empty(self):
        assert extract_tool_calls("not a list") == []
        assert extract_tool_calls(None) == []

    def test_missing_function_key(self):
        raw = [{"id": "call_2"}]
        calls = extract_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].tool_name == ""
        assert calls[0].arguments == {}

    def test_args_as_dict(self):
        raw = [{"function": {"name": "f", "arguments": {"k": "v"}}}]
        calls = extract_tool_calls(raw)
        assert calls[0].arguments == {"k": "v"}

    def test_args_as_json_string(self):
        raw = [{"function": {"name": "f", "arguments": '{"k": "v"}'}}]
        calls = extract_tool_calls(raw)
        assert calls[0].arguments == {"k": "v"}

    def test_invalid_args_string(self):
        raw = [{"function": {"name": "f", "arguments": "not json"}}]
        calls = extract_tool_calls(raw)
        assert calls[0].arguments == {}


# ---------------------------------------------------------------------------
# normalization.py -- finalize_stream_tool_calls
# ---------------------------------------------------------------------------


class TestFinalizeStreamToolCalls:
    def test_empty(self):
        assert finalize_stream_tool_calls({}) == []

    def test_single_buffer(self):
        buffers = {
            0: {
                "id": "c1",
                "name": "do_thing",
                "args_parts": ['{"a":', '"b"}'],
            }
        }
        calls = finalize_stream_tool_calls(buffers)
        assert len(calls) == 1
        assert calls[0] == ToolCall(id="c1", tool_name="do_thing", arguments={"a": "b"})

    def test_multiple_sorted_by_index(self):
        buffers = {
            2: {"id": "c3", "name": "third", "args_parts": ['{}']},
            0: {"id": "c1", "name": "first", "args_parts": ['{}']},
            1: {"id": "c2", "name": "second", "args_parts": ['{}']},
        }
        calls = finalize_stream_tool_calls(buffers)
        assert [c.tool_name for c in calls] == ["first", "second", "third"]

    def test_missing_fields(self):
        buffers = {0: {}}
        calls = finalize_stream_tool_calls(buffers)
        assert len(calls) == 1
        assert calls[0].id is None
        assert calls[0].tool_name == ""
        assert calls[0].arguments == {}

    def test_invalid_json_args_parts(self):
        buffers = {0: {"name": "f", "args_parts": ["not", "json"]}}
        calls = finalize_stream_tool_calls(buffers)
        assert calls[0].arguments == {}


# ---------------------------------------------------------------------------
# normalization.py -- get_attr
# ---------------------------------------------------------------------------


class TestGetAttr:
    def test_existing_attr(self):
        class Obj:
            x = 42

        assert get_attr(Obj(), "x") == 42

    def test_missing_attr_returns_none(self):
        class Obj:
            pass

        assert get_attr(Obj(), "missing") is None


# ---------------------------------------------------------------------------
# normalization.py -- get_attr_str
# ---------------------------------------------------------------------------


class TestGetAttrStr:
    def test_string_attr_returned(self):
        class Obj:
            name = "hello"

        assert get_attr_str(Obj(), "name") == "hello"

    def test_non_string_attr_returns_none(self):
        class Obj:
            value = 42

        assert get_attr_str(Obj(), "value") is None

    def test_missing_attr_returns_none(self):
        class Obj:
            pass

        assert get_attr_str(Obj(), "nope") is None


# ---------------------------------------------------------------------------
# transport.py -- collect_headers
# ---------------------------------------------------------------------------


class TestCollectHeaders:
    def test_non_dict_existing_headers_ignored(self):
        result = collect_headers("not a dict", idempotency_key=None, metadata=None)
        assert result == {}

    def test_dict_existing_headers_copied(self):
        result = collect_headers(
            {"Authorization": "Bearer tok"},
            idempotency_key=None,
            metadata=None,
        )
        assert result == {"Authorization": "Bearer tok"}

    def test_non_string_keys_and_values_skipped(self):
        result = collect_headers(
            {123: "bad_key", "good": 456, "ok": "ok"},
            idempotency_key=None,
            metadata=None,
        )
        assert result == {"ok": "ok"}

    def test_idempotency_key_string_sets_header(self):
        result = collect_headers(None, idempotency_key="abc-123", metadata=None)
        assert result["Idempotency-Key"] == "abc-123"

    def test_empty_idempotency_key_no_header(self):
        result = collect_headers(None, idempotency_key="", metadata=None)
        assert "Idempotency-Key" not in result

    def test_none_idempotency_key_no_header(self):
        result = collect_headers(None, idempotency_key=None, metadata=None)
        assert "Idempotency-Key" not in result

    def test_metadata_afk_request_id_sets_header(self):
        result = collect_headers(
            None,
            idempotency_key=None,
            metadata={"afk_request_id": "req-1"},
        )
        assert result["X-Request-Id"] == "req-1"

    def test_metadata_without_afk_request_id_no_header(self):
        result = collect_headers(None, idempotency_key=None, metadata={"other": "val"})
        assert "X-Request-Id" not in result

    def test_existing_idempotency_key_not_overwritten(self):
        result = collect_headers(
            {"Idempotency-Key": "original"},
            idempotency_key="new",
            metadata=None,
        )
        assert result["Idempotency-Key"] == "original"
