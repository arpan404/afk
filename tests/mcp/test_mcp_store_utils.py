"""
Tests for afk.mcp.store.utils and afk.mcp.store.types.
"""

import pytest

from afk.mcp.store.types import (
    MCPRemoteCallError,
    MCPRemoteProtocolError,
    MCPRemoteTool,
    MCPServerRef,
    MCPServerResolutionError,
    MCPStoreError,
)
from afk.mcp.store.utils import (
    _extract_mcp_text,
    _qualified_tool_name,
    _sanitize_name,
    _validate_http_url,
    normalize_json_schema,
    normalize_remote_tools,
    resolve_server_ref,
)


# ── _sanitize_name ──────────────────────────────────────────────────────────


class TestSanitizeName:
    def test_letters_and_underscores_preserved(self):
        assert _sanitize_name("hello_world") == "hello_world"

    def test_special_chars_replaced_with_underscore(self):
        assert _sanitize_name("hello-world.v2") == "hello_world_v2"

    def test_leading_trailing_underscores_stripped(self):
        assert _sanitize_name("__name__") == "name"

    def test_empty_result_falls_back_to_mcp(self):
        assert _sanitize_name("---") == "mcp"

    def test_already_clean_name(self):
        assert _sanitize_name("my_tool") == "my_tool"

    def test_pure_digits(self):
        assert _sanitize_name("123") == "123"

    def test_mixed_special_chars(self):
        assert _sanitize_name("a@b#c$d") == "a_b_c_d"

    def test_empty_string_falls_back_to_mcp(self):
        assert _sanitize_name("") == "mcp"


# ── _validate_http_url ──────────────────────────────────────────────────────


class TestValidateHttpUrl:
    def test_valid_http_url_passes(self):
        url = "http://localhost:8000"
        assert _validate_http_url(url) == url

    def test_valid_https_url_passes(self):
        url = "https://example.com/mcp"
        assert _validate_http_url(url) == url

    def test_non_http_scheme_raises(self):
        with pytest.raises(MCPServerResolutionError, match="scheme must be http or https"):
            _validate_http_url("ftp://host")

    def test_missing_netloc_raises(self):
        with pytest.raises(MCPServerResolutionError, match="must include network location"):
            _validate_http_url("http://")

    def test_websocket_scheme_raises(self):
        with pytest.raises(MCPServerResolutionError):
            _validate_http_url("ws://localhost:8000")

    def test_empty_scheme_raises(self):
        with pytest.raises(MCPServerResolutionError):
            _validate_http_url("://localhost")


# ── _qualified_tool_name ─────────────────────────────────────────────────────


class TestQualifiedToolName:
    def test_prefix_tools_true_default(self):
        server = MCPServerRef(name="prefix", url="http://localhost:8000")
        assert _qualified_tool_name(server, "tool") == "prefix__tool"

    def test_prefix_tools_false_returns_tool_name_as_is(self):
        server = MCPServerRef(
            name="prefix", url="http://localhost:8000", prefix_tools=False
        )
        assert _qualified_tool_name(server, "tool") == "tool"

    def test_tool_name_prefix_overrides_server_name(self):
        server = MCPServerRef(
            name="prefix",
            url="http://localhost:8000",
            tool_name_prefix="custom",
        )
        assert _qualified_tool_name(server, "tool") == "custom__tool"

    def test_prefix_sanitized(self):
        server = MCPServerRef(name="my-server.v2", url="http://localhost:8000")
        result = _qualified_tool_name(server, "run")
        assert result == "my_server_v2__run"

    def test_tool_name_sanitized(self):
        server = MCPServerRef(name="srv", url="http://localhost:8000")
        result = _qualified_tool_name(server, "some-tool.v1")
        assert result == "srv__some_tool_v1"


# ── _extract_mcp_text ───────────────────────────────────────────────────────


class TestExtractMcpText:
    def test_string_returns_string(self):
        assert _extract_mcp_text("hello") == "hello"

    def test_list_of_dicts_with_text_key_joined(self):
        content = [{"text": "line1"}, {"text": "line2"}]
        assert _extract_mcp_text(content) == "line1\nline2"

    def test_empty_list_returns_none(self):
        assert _extract_mcp_text([]) is None

    def test_non_list_non_string_returns_none(self):
        assert _extract_mcp_text(42) is None
        assert _extract_mcp_text(None) is None

    def test_list_with_no_text_entries_returns_none(self):
        content = [{"image": "data"}, {"type": "blob"}]
        assert _extract_mcp_text(content) is None

    def test_list_with_mixed_entries(self):
        content = [{"text": "hello"}, {"image": "data"}, {"text": "world"}]
        assert _extract_mcp_text(content) == "hello\nworld"

    def test_list_with_non_dict_entries_skipped(self):
        content = ["not_a_dict", {"text": "ok"}]
        assert _extract_mcp_text(content) == "ok"

    def test_list_with_non_string_text_value_skipped(self):
        content = [{"text": 123}, {"text": "valid"}]
        assert _extract_mcp_text(content) == "valid"


# ── resolve_server_ref ───────────────────────────────────────────────────────


class TestResolveServerRef:
    def test_mcp_server_ref_passthrough(self):
        ref = MCPServerRef(name="srv", url="http://localhost:8000")
        assert resolve_server_ref(ref) is ref

    def test_string_name_equals_url(self):
        result = resolve_server_ref("myserver=http://localhost:8000")
        assert isinstance(result, MCPServerRef)
        assert result.name == "myserver"
        assert result.url == "http://localhost:8000"

    def test_string_http_url_derives_name_from_host(self):
        result = resolve_server_ref("http://example.com:9000")
        assert isinstance(result, MCPServerRef)
        assert "example" in result.name
        assert result.url == "http://example.com:9000"

    def test_empty_string_raises(self):
        with pytest.raises(MCPServerResolutionError, match="cannot be empty"):
            resolve_server_ref("")

    def test_whitespace_string_raises(self):
        with pytest.raises(MCPServerResolutionError, match="cannot be empty"):
            resolve_server_ref("   ")

    def test_non_http_string_raises(self):
        with pytest.raises(MCPServerResolutionError, match="http"):
            resolve_server_ref("just-a-name")

    def test_dict_with_url(self):
        result = resolve_server_ref({"url": "http://localhost:8000", "name": "myname"})
        assert isinstance(result, MCPServerRef)
        assert result.name == "myname"
        assert result.url == "http://localhost:8000"

    def test_dict_missing_url_raises(self):
        with pytest.raises(MCPServerResolutionError, match="non-empty 'url'"):
            resolve_server_ref({"name": "only_name"})

    def test_dict_empty_url_raises(self):
        with pytest.raises(MCPServerResolutionError, match="non-empty 'url'"):
            resolve_server_ref({"url": "  "})

    def test_dict_with_invalid_timeout_raises(self):
        with pytest.raises(MCPServerResolutionError, match="timeout_s"):
            resolve_server_ref({"url": "http://localhost:8000", "timeout_s": -5})

    def test_dict_with_zero_timeout_raises(self):
        with pytest.raises(MCPServerResolutionError, match="timeout_s"):
            resolve_server_ref({"url": "http://localhost:8000", "timeout_s": 0})

    def test_dict_with_string_timeout_raises(self):
        with pytest.raises(MCPServerResolutionError, match="timeout_s"):
            resolve_server_ref({"url": "http://localhost:8000", "timeout_s": "fast"})

    def test_dict_with_non_dict_headers_raises(self):
        with pytest.raises(MCPServerResolutionError, match="headers.*dict"):
            resolve_server_ref(
                {"url": "http://localhost:8000", "headers": ["not", "dict"]}
            )

    def test_dict_with_valid_headers(self):
        result = resolve_server_ref(
            {
                "url": "http://localhost:8000",
                "headers": {"Authorization": "Bearer tok"},
            }
        )
        assert result.headers == {"Authorization": "Bearer tok"}

    def test_dict_derives_name_from_url_when_name_missing(self):
        result = resolve_server_ref({"url": "http://example.com:9000"})
        assert "example" in result.name

    def test_dict_with_prefix_tools_false(self):
        result = resolve_server_ref(
            {"url": "http://localhost:8000", "prefix_tools": False}
        )
        assert result.prefix_tools is False

    def test_dict_with_tool_name_prefix(self):
        result = resolve_server_ref(
            {"url": "http://localhost:8000", "tool_name_prefix": "custom"}
        )
        assert result.tool_name_prefix == "custom"

    def test_dict_with_non_string_tool_name_prefix_raises(self):
        with pytest.raises(MCPServerResolutionError, match="tool_name_prefix.*string"):
            resolve_server_ref(
                {"url": "http://localhost:8000", "tool_name_prefix": 123}
            )

    def test_unsupported_type_raises(self):
        with pytest.raises(MCPServerResolutionError, match="Unsupported"):
            resolve_server_ref(42)

    def test_unsupported_type_list_raises(self):
        with pytest.raises(MCPServerResolutionError, match="Unsupported"):
            resolve_server_ref(["http://localhost"])


# ── normalize_remote_tools ───────────────────────────────────────────────────


class TestNormalizeRemoteTools:
    def _make_server(self, **kwargs):
        defaults = {"name": "srv", "url": "http://localhost:8000"}
        defaults.update(kwargs)
        return MCPServerRef(**defaults)

    def test_valid_tools_list_normalized(self):
        server = self._make_server()
        tools = [
            {
                "name": "do_thing",
                "description": "Does a thing",
                "inputSchema": {"type": "object", "properties": {"x": {"type": "int"}}},
            }
        ]
        result = normalize_remote_tools(server, tools)
        assert len(result) == 1
        assert isinstance(result[0], MCPRemoteTool)
        assert result[0].name == "do_thing"
        assert result[0].description == "Does a thing"
        assert result[0].qualified_name == "srv__do_thing"
        assert result[0].server_name == "srv"

    def test_non_list_raises_protocol_error(self):
        server = self._make_server()
        with pytest.raises(MCPRemoteProtocolError, match="missing tools list"):
            normalize_remote_tools(server, "not_a_list")

    def test_dict_without_name_skipped(self):
        server = self._make_server()
        tools = [{"description": "no name here"}]
        result = normalize_remote_tools(server, tools)
        assert len(result) == 0

    def test_empty_name_skipped(self):
        server = self._make_server()
        tools = [{"name": "  "}]
        result = normalize_remote_tools(server, tools)
        assert len(result) == 0

    def test_missing_input_schema_defaults_to_object(self):
        server = self._make_server()
        tools = [{"name": "simple_tool"}]
        result = normalize_remote_tools(server, tools)
        assert len(result) == 1
        assert result[0].input_schema == {"type": "object"}

    def test_non_dict_entries_skipped(self):
        server = self._make_server()
        tools = ["not_a_dict", 42, {"name": "valid"}]
        result = normalize_remote_tools(server, tools)
        assert len(result) == 1
        assert result[0].name == "valid"

    def test_missing_description_falls_back_to_name(self):
        server = self._make_server()
        tools = [{"name": "my_tool"}]
        result = normalize_remote_tools(server, tools)
        assert result[0].description == "my_tool"

    def test_non_string_description_falls_back_to_name(self):
        server = self._make_server()
        tools = [{"name": "my_tool", "description": 42}]
        result = normalize_remote_tools(server, tools)
        assert result[0].description == "my_tool"

    def test_non_dict_input_schema_defaults(self):
        server = self._make_server()
        tools = [{"name": "tool", "inputSchema": "not_a_dict"}]
        result = normalize_remote_tools(server, tools)
        assert result[0].input_schema == {"type": "object"}

    def test_prefix_tools_false_uses_plain_name(self):
        server = self._make_server(prefix_tools=False)
        tools = [{"name": "my_tool"}]
        result = normalize_remote_tools(server, tools)
        assert result[0].qualified_name == "my_tool"

    def test_empty_list_returns_empty(self):
        server = self._make_server()
        result = normalize_remote_tools(server, [])
        assert result == []


# ── normalize_json_schema ────────────────────────────────────────────────────


class TestNormalizeJsonSchema:
    def test_non_dict_returns_minimal_schema(self):
        result = normalize_json_schema("not_a_dict")
        assert result == {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    def test_forces_type_to_object(self):
        result = normalize_json_schema({"type": "string"})
        assert result["type"] == "object"

    def test_normalizes_properties(self):
        result = normalize_json_schema(
            {"properties": {"x": {"type": "string"}, "y": {"type": "integer"}}}
        )
        assert "x" in result["properties"]
        assert "y" in result["properties"]

    def test_non_dict_properties_replaced(self):
        result = normalize_json_schema({"properties": "not_a_dict"})
        assert result["properties"] == {}

    def test_non_dict_property_value_replaced_with_empty_dict(self):
        result = normalize_json_schema({"properties": {"x": "not_a_dict"}})
        assert result["properties"]["x"] == {}

    def test_normalizes_required(self):
        result = normalize_json_schema(
            {
                "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
                "required": ["a"],
            }
        )
        assert result["required"] == ["a"]

    def test_invalid_required_items_filtered(self):
        result = normalize_json_schema(
            {
                "properties": {"a": {"type": "string"}},
                "required": [123, None, "a"],
            }
        )
        assert result["required"] == ["a"]

    def test_required_keys_not_in_properties_filtered(self):
        result = normalize_json_schema(
            {
                "properties": {"a": {"type": "string"}},
                "required": ["a", "nonexistent"],
            }
        )
        assert result["required"] == ["a"]

    def test_non_list_required_becomes_empty(self):
        result = normalize_json_schema({"required": "not_a_list"})
        assert result["required"] == []

    def test_normalizes_additional_properties_bool(self):
        result = normalize_json_schema({"additionalProperties": True})
        assert result["additionalProperties"] is True

    def test_normalizes_additional_properties_dict(self):
        schema = {"additionalProperties": {"type": "string"}}
        result = normalize_json_schema(schema)
        assert result["additionalProperties"] == {"type": "string"}

    def test_invalid_additional_properties_set_to_false(self):
        result = normalize_json_schema({"additionalProperties": "yes"})
        assert result["additionalProperties"] is False

    def test_missing_additional_properties_set_to_false(self):
        result = normalize_json_schema({})
        assert result["additionalProperties"] is False

    def test_preserves_extra_keys(self):
        result = normalize_json_schema({"description": "test schema"})
        assert result.get("description") == "test schema"


# ── MCPServerRef dataclass ───────────────────────────────────────────────────


class TestMCPServerRef:
    def test_default_values(self):
        ref = MCPServerRef(name="test", url="http://localhost")
        assert ref.headers == {}
        assert ref.timeout_s == 20.0
        assert ref.prefix_tools is True
        assert ref.tool_name_prefix is None

    def test_custom_values(self):
        ref = MCPServerRef(
            name="custom",
            url="https://example.com",
            headers={"X-Key": "val"},
            timeout_s=10.0,
            prefix_tools=False,
            tool_name_prefix="myprefix",
        )
        assert ref.name == "custom"
        assert ref.url == "https://example.com"
        assert ref.headers == {"X-Key": "val"}
        assert ref.timeout_s == 10.0
        assert ref.prefix_tools is False
        assert ref.tool_name_prefix == "myprefix"

    def test_frozen(self):
        ref = MCPServerRef(name="test", url="http://localhost")
        with pytest.raises(AttributeError):
            ref.name = "changed"


# ── MCP error hierarchy ─────────────────────────────────────────────────────


class TestMCPErrorHierarchy:
    def test_mcp_store_error_is_runtime_error(self):
        assert issubclass(MCPStoreError, RuntimeError)

    def test_mcp_server_resolution_error_is_mcp_store_error(self):
        assert issubclass(MCPServerResolutionError, MCPStoreError)

    def test_mcp_remote_protocol_error_is_mcp_store_error(self):
        assert issubclass(MCPRemoteProtocolError, MCPStoreError)

    def test_mcp_remote_call_error_is_mcp_store_error(self):
        assert issubclass(MCPRemoteCallError, MCPStoreError)

    def test_mcp_server_resolution_error_is_runtime_error(self):
        assert issubclass(MCPServerResolutionError, RuntimeError)

    def test_mcp_remote_protocol_error_is_runtime_error(self):
        assert issubclass(MCPRemoteProtocolError, RuntimeError)

    def test_mcp_remote_call_error_is_runtime_error(self):
        assert issubclass(MCPRemoteCallError, RuntimeError)

    def test_instances_catchable_as_mcp_store_error(self):
        with pytest.raises(MCPStoreError):
            raise MCPServerResolutionError("test")

    def test_instances_catchable_as_runtime_error(self):
        with pytest.raises(RuntimeError):
            raise MCPRemoteProtocolError("test")
