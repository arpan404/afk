"""Tests for the LLM factory legacy shims and the LLM error hierarchy."""

from __future__ import annotations

import pytest

from afk.llms.factory import (
    register_llm_adapter,
    available_llm_adapters,
    create_llm,
    create_llm_from_env,
)
from afk.llms.errors import (
    LLMError,
    LLMTimeoutError,
    LLMRetryableError,
    LLMInvalidResponseError,
    LLMConfigurationError,
    LLMCapabilityError,
    LLMCancelledError,
    LLMInterruptedError,
    LLMSessionError,
    LLMSessionPausedError,
)
from afk.llms.types import Usage, LLMResponse, LLMRequest, ToolCall


# ---------------------------------------------------------------------------
# Factory legacy shims -- all must raise LLMConfigurationError
# ---------------------------------------------------------------------------


class TestFactoryLegacyShims:
    """Every legacy factory function should raise LLMConfigurationError."""

    def test_register_llm_adapter_raises(self):
        with pytest.raises(LLMConfigurationError, match="Legacy factory APIs are removed"):
            register_llm_adapter()

    def test_available_llm_adapters_raises(self):
        with pytest.raises(LLMConfigurationError, match="Legacy factory APIs are removed"):
            available_llm_adapters()

    def test_create_llm_raises(self):
        with pytest.raises(LLMConfigurationError, match="Legacy factory APIs are removed"):
            create_llm()

    def test_create_llm_from_env_raises(self):
        with pytest.raises(LLMConfigurationError, match="Legacy factory APIs are removed"):
            create_llm_from_env()

    # -- arbitrary args/kwargs are accepted but still raise --

    def test_register_llm_adapter_with_args(self):
        with pytest.raises(LLMConfigurationError):
            register_llm_adapter("adapter_name", key="value")

    def test_available_llm_adapters_with_args(self):
        with pytest.raises(LLMConfigurationError):
            available_llm_adapters("some", "args", flag=True)

    def test_create_llm_with_args(self):
        with pytest.raises(LLMConfigurationError):
            create_llm("openai", model="gpt-4", temperature=0.7)

    def test_create_llm_from_env_with_args(self):
        with pytest.raises(LLMConfigurationError):
            create_llm_from_env("MY_KEY", extra=42)

    # -- error message contains upgrade hint --

    def test_error_message_mentions_alternative(self):
        with pytest.raises(LLMConfigurationError, match="LLMBuilder"):
            create_llm()


# ---------------------------------------------------------------------------
# Error hierarchy -- class relationships
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    """Verify inheritance chain and that all errors descend from LLMError."""

    def test_llm_error_is_exception(self):
        assert issubclass(LLMError, Exception)

    def test_llm_timeout_error_is_llm_error(self):
        assert issubclass(LLMTimeoutError, LLMError)

    def test_llm_retryable_error_is_llm_error(self):
        assert issubclass(LLMRetryableError, LLMError)

    def test_llm_invalid_response_error_is_llm_error(self):
        assert issubclass(LLMInvalidResponseError, LLMError)

    def test_llm_configuration_error_is_llm_error(self):
        assert issubclass(LLMConfigurationError, LLMError)

    def test_llm_capability_error_is_llm_error(self):
        assert issubclass(LLMCapabilityError, LLMError)

    def test_llm_cancelled_error_is_llm_error(self):
        assert issubclass(LLMCancelledError, LLMError)

    def test_llm_interrupted_error_is_llm_error(self):
        assert issubclass(LLMInterruptedError, LLMError)

    def test_llm_session_error_is_llm_error(self):
        assert issubclass(LLMSessionError, LLMError)

    def test_llm_session_paused_error_is_llm_session_error(self):
        assert issubclass(LLMSessionPausedError, LLMSessionError)

    def test_llm_session_paused_error_is_also_llm_error(self):
        assert issubclass(LLMSessionPausedError, LLMError)


# ---------------------------------------------------------------------------
# Error hierarchy -- message storage
# ---------------------------------------------------------------------------


class TestErrorMessages:
    """Each error class stores the message correctly via str()."""

    @pytest.mark.parametrize(
        "cls",
        [
            LLMError,
            LLMTimeoutError,
            LLMRetryableError,
            LLMInvalidResponseError,
            LLMConfigurationError,
            LLMCapabilityError,
            LLMCancelledError,
            LLMInterruptedError,
            LLMSessionError,
            LLMSessionPausedError,
        ],
    )
    def test_message_stored(self, cls):
        msg = f"test message for {cls.__name__}"
        err = cls(msg)
        assert str(err) == msg
        assert err.args[0] == msg


# ---------------------------------------------------------------------------
# Error hierarchy -- catchability by parent class
# ---------------------------------------------------------------------------


class TestErrorCatchability:
    """Raising a subclass is catchable by the parent handler."""

    def test_timeout_caught_by_llm_error(self):
        with pytest.raises(LLMError):
            raise LLMTimeoutError("timeout")

    def test_retryable_caught_by_llm_error(self):
        with pytest.raises(LLMError):
            raise LLMRetryableError("retry")

    def test_invalid_response_caught_by_llm_error(self):
        with pytest.raises(LLMError):
            raise LLMInvalidResponseError("bad response")

    def test_configuration_caught_by_llm_error(self):
        with pytest.raises(LLMError):
            raise LLMConfigurationError("bad config")

    def test_capability_caught_by_llm_error(self):
        with pytest.raises(LLMError):
            raise LLMCapabilityError("not supported")

    def test_cancelled_caught_by_llm_error(self):
        with pytest.raises(LLMError):
            raise LLMCancelledError("cancelled")

    def test_interrupted_caught_by_llm_error(self):
        with pytest.raises(LLMError):
            raise LLMInterruptedError("interrupted")

    def test_session_error_caught_by_llm_error(self):
        with pytest.raises(LLMError):
            raise LLMSessionError("session")

    def test_session_paused_caught_by_session_error(self):
        with pytest.raises(LLMSessionError):
            raise LLMSessionPausedError("paused")

    def test_session_paused_caught_by_llm_error(self):
        with pytest.raises(LLMError):
            raise LLMSessionPausedError("paused")

    def test_all_caught_by_exception(self):
        with pytest.raises(Exception):
            raise LLMTimeoutError("generic")


# ---------------------------------------------------------------------------
# LLM Types spot checks
# ---------------------------------------------------------------------------


class TestUsageType:
    """Usage dataclass defaults and structure."""

    def test_default_none_fields(self):
        u = Usage()
        assert u.input_tokens is None
        assert u.output_tokens is None
        assert u.total_tokens is None

    def test_explicit_values(self):
        u = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        assert u.input_tokens == 10
        assert u.output_tokens == 20
        assert u.total_tokens == 30

    def test_frozen(self):
        u = Usage()
        with pytest.raises(AttributeError):
            u.input_tokens = 5  # type: ignore[misc]


class TestLLMResponseType:
    """LLMResponse dataclass defaults and structure."""

    def test_text_field(self):
        r = LLMResponse(text="hello")
        assert r.text == "hello"

    def test_tool_calls_default_empty(self):
        r = LLMResponse(text="")
        assert r.tool_calls == []
        assert isinstance(r.tool_calls, list)

    def test_usage_default(self):
        r = LLMResponse(text="")
        assert isinstance(r.usage, Usage)
        assert r.usage.input_tokens is None

    def test_frozen(self):
        r = LLMResponse(text="x")
        with pytest.raises(AttributeError):
            r.text = "y"  # type: ignore[misc]


class TestLLMRequestType:
    """LLMRequest dataclass has model and messages fields."""

    def test_model_field(self):
        req = LLMRequest(model="gpt-4")
        assert req.model == "gpt-4"

    def test_messages_default_empty(self):
        req = LLMRequest(model="gpt-4")
        assert req.messages == []

    def test_frozen(self):
        req = LLMRequest(model="gpt-4")
        with pytest.raises(AttributeError):
            req.model = "gpt-3"  # type: ignore[misc]


class TestToolCallType:
    """ToolCall dataclass has id, tool_name, and arguments fields."""

    def test_default_values(self):
        tc = ToolCall()
        assert tc.id is None
        assert tc.tool_name == ""
        assert tc.arguments == {}

    def test_explicit_values(self):
        tc = ToolCall(id="call_1", tool_name="my_fn", arguments={"x": 1})
        assert tc.id == "call_1"
        assert tc.tool_name == "my_fn"
        assert tc.arguments == {"x": 1}

    def test_frozen(self):
        tc = ToolCall()
        with pytest.raises(AttributeError):
            tc.id = "new"  # type: ignore[misc]
