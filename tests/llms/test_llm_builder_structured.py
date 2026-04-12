"""
Comprehensive tests for:
  - LLMBuilder fluent API and profile application
  - PROFILES dictionary structure and values
  - Structured output helpers (json_system_prompt, parse_and_validate_json, make_repair_prompt)
  - LLMLifecycleEvent dataclass and LLMLifecycleEventType literal
  - MiddlewareStack default and parameterized construction
"""

from __future__ import annotations

import json
import os

import pytest
from pydantic import BaseModel

from afk.llms.builder import LLMBuilder
from afk.llms.errors import LLMInvalidResponseError
from afk.llms.middleware import MiddlewareStack
from afk.llms.observability import LLMLifecycleEvent
from afk.llms.profiles import PROFILES
from afk.llms.runtime.contracts import (
    CachePolicy,
    CircuitBreakerPolicy,
    CoalescingPolicy,
    HedgingPolicy,
    RateLimitPolicy,
    RetryPolicy,
    TimeoutPolicy,
)
from afk.llms.settings import LLMSettings
from afk.llms.structured import (
    json_system_prompt,
    make_repair_prompt,
    parse_and_validate_json,
)


# ============================== Helpers ==============================

# All AFK_LLM_* env vars that LLMSettings.from_env() reads.
_ALL_AFK_ENV_VARS = (
    "AFK_LLM_PROVIDER",
    "AFK_LLM_MODEL",
    "AFK_EMBED_MODEL",
    "AFK_LLM_API_BASE_URL",
    "AFK_LLM_API_KEY",
    "AFK_LLM_TIMEOUT_S",
    "AFK_LLM_MAX_RETRIES",
    "AFK_LLM_BACKOFF_BASE_S",
    "AFK_LLM_BACKOFF_JITTER_S",
    "AFK_LLM_JSON_MAX_RETRIES",
    "AFK_LLM_MAX_INPUT_CHARS",
    "AFK_LLM_STREAM_IDLE_TIMEOUT_S",
)


@pytest.fixture(autouse=True)
def _clean_env():
    """Remove all AFK env vars before and after each test to guarantee isolation."""
    saved = {}
    for var in _ALL_AFK_ENV_VARS:
        saved[var] = os.environ.pop(var, None)
    yield
    for var in _ALL_AFK_ENV_VARS:
        if saved[var] is not None:
            os.environ[var] = saved[var]
        else:
            os.environ.pop(var, None)


class _SampleModel(BaseModel):
    """Simple Pydantic model used by structured output tests."""

    name: str
    value: int


class _OtherModel(BaseModel):
    """Alternative model for schema-mismatch tests."""

    title: str
    count: int
    active: bool


# ============================== LLMBuilder ==============================


class TestLLMBuilderConstructor:
    """Builder constructor should set clean defaults."""

    def test_provider_is_none(self):
        b = LLMBuilder()
        assert b._provider is None

    def test_settings_from_env(self):
        b = LLMBuilder()
        assert isinstance(b._settings, LLMSettings)

    def test_settings_default_provider(self):
        b = LLMBuilder()
        assert b._settings.default_provider == "litellm"

    def test_settings_default_model(self):
        b = LLMBuilder()
        assert b._settings.default_model == "gpt-4.1-mini"

    def test_provider_settings_empty(self):
        b = LLMBuilder()
        assert b._provider_settings == {}

    def test_middlewares_none(self):
        b = LLMBuilder()
        assert b._middlewares is None

    def test_observers_none(self):
        b = LLMBuilder()
        assert b._observers is None

    def test_cache_backend_none(self):
        b = LLMBuilder()
        assert b._cache_backend is None

    def test_router_none(self):
        b = LLMBuilder()
        assert b._router is None

    def test_retry_policy_none(self):
        b = LLMBuilder()
        assert b._retry_policy is None

    def test_timeout_policy_none(self):
        b = LLMBuilder()
        assert b._timeout_policy is None

    def test_rate_limit_policy_none(self):
        b = LLMBuilder()
        assert b._rate_limit_policy is None

    def test_breaker_policy_none(self):
        b = LLMBuilder()
        assert b._breaker_policy is None

    def test_hedging_policy_none(self):
        b = LLMBuilder()
        assert b._hedging_policy is None

    def test_cache_policy_none(self):
        b = LLMBuilder()
        assert b._cache_policy is None

    def test_coalescing_policy_none(self):
        b = LLMBuilder()
        assert b._coalescing_policy is None


class TestLLMBuilderProvider:
    """`.provider()` strips, lowercases, and stores the provider id."""

    def test_sets_provider_lowercase(self):
        b = LLMBuilder().provider("OpenAI")
        assert b._provider == "openai"

    def test_strips_whitespace(self):
        b = LLMBuilder().provider("  Anthropic  ")
        assert b._provider == "anthropic"

    def test_already_lowercase(self):
        b = LLMBuilder().provider("litellm")
        assert b._provider == "litellm"

    def test_mixed_case_with_spaces(self):
        b = LLMBuilder().provider("  LiteLLM  ")
        assert b._provider == "litellm"


class TestLLMBuilderModel:
    """`.model()` replaces settings with an updated default_model."""

    def test_updates_default_model(self):
        b = LLMBuilder().model("gpt-4")
        assert b._settings.default_model == "gpt-4"

    def test_preserves_other_settings(self):
        b = LLMBuilder().model("gpt-4")
        assert b._settings.default_provider == "litellm"
        assert b._settings.timeout_s == 30.0
        assert b._settings.max_retries == 3

    def test_chained_model_calls(self):
        b = LLMBuilder().model("gpt-4").model("claude-sonnet-4-20250514")
        assert b._settings.default_model == "claude-sonnet-4-20250514"


class TestLLMBuilderSettings:
    """`.settings()` replaces the entire settings object."""

    def test_replaces_settings(self):
        custom = LLMSettings(default_provider="openai", default_model="gpt-4o")
        b = LLMBuilder().settings(custom)
        assert b._settings is custom
        assert b._settings.default_provider == "openai"
        assert b._settings.default_model == "gpt-4o"


class TestLLMBuilderProfile:
    """`.profile()` applies a named profile's runtime policies."""

    def test_development_profile_sets_all_policies(self):
        b = LLMBuilder().profile("development")
        assert isinstance(b._retry_policy, RetryPolicy)
        assert isinstance(b._timeout_policy, TimeoutPolicy)
        assert isinstance(b._rate_limit_policy, RateLimitPolicy)
        assert isinstance(b._breaker_policy, CircuitBreakerPolicy)
        assert isinstance(b._hedging_policy, HedgingPolicy)
        assert isinstance(b._cache_policy, CachePolicy)
        assert isinstance(b._coalescing_policy, CoalescingPolicy)

    def test_development_profile_retry_values(self):
        b = LLMBuilder().profile("development")
        assert b._retry_policy.max_retries == 1
        assert b._retry_policy.backoff_base_s == 0.2
        assert b._retry_policy.backoff_jitter_s == 0.05

    def test_development_profile_hedging_disabled(self):
        b = LLMBuilder().profile("development")
        assert b._hedging_policy.enabled is False

    def test_development_profile_cache_disabled(self):
        b = LLMBuilder().profile("development")
        assert b._cache_policy.enabled is False

    def test_production_profile_sets_all_policies(self):
        b = LLMBuilder().profile("production")
        assert isinstance(b._retry_policy, RetryPolicy)
        assert isinstance(b._timeout_policy, TimeoutPolicy)
        assert isinstance(b._rate_limit_policy, RateLimitPolicy)
        assert isinstance(b._breaker_policy, CircuitBreakerPolicy)
        assert isinstance(b._hedging_policy, HedgingPolicy)
        assert isinstance(b._cache_policy, CachePolicy)
        assert isinstance(b._coalescing_policy, CoalescingPolicy)

    def test_production_profile_retry_values(self):
        b = LLMBuilder().profile("production")
        assert b._retry_policy.max_retries == 3
        assert b._retry_policy.backoff_base_s == 0.5
        assert b._retry_policy.backoff_jitter_s == 0.15

    def test_production_profile_timeout_values(self):
        b = LLMBuilder().profile("production")
        assert b._timeout_policy.request_timeout_s == 30.0
        assert b._timeout_policy.stream_idle_timeout_s == 45.0

    def test_production_profile_rate_limit_values(self):
        b = LLMBuilder().profile("production")
        assert b._rate_limit_policy.requests_per_second == 20.0
        assert b._rate_limit_policy.burst == 40

    def test_production_profile_breaker_values(self):
        b = LLMBuilder().profile("production")
        assert b._breaker_policy.failure_threshold == 5
        assert b._breaker_policy.cooldown_s == 30.0
        assert b._breaker_policy.half_open_max_calls == 1

    def test_production_profile_hedging_disabled(self):
        b = LLMBuilder().profile("production")
        assert b._hedging_policy.enabled is False

    def test_production_profile_cache_disabled(self):
        b = LLMBuilder().profile("production")
        assert b._cache_policy.enabled is False
        assert b._cache_policy.ttl_s == 30.0

    def test_production_profile_coalescing_enabled(self):
        b = LLMBuilder().profile("production")
        assert b._coalescing_policy.enabled is True

    def test_unknown_profile_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown llm profile"):
            LLMBuilder().profile("unknown")

    def test_unknown_profile_preserves_exact_name_in_message(self):
        with pytest.raises(ValueError, match="nonexistent_profile"):
            LLMBuilder().profile("nonexistent_profile")

    def test_profile_strips_and_lowercases(self):
        b = LLMBuilder().profile("  Production  ")
        assert b._retry_policy.max_retries == 3


class TestLLMBuilderForAgentRuntime:
    """`.for_agent_runtime()` is equivalent to `.profile("production")`."""

    def test_same_retry_policy(self):
        agent_builder = LLMBuilder().for_agent_runtime()
        prod_builder = LLMBuilder().profile("production")
        assert agent_builder._retry_policy == prod_builder._retry_policy

    def test_same_timeout_policy(self):
        agent_builder = LLMBuilder().for_agent_runtime()
        prod_builder = LLMBuilder().profile("production")
        assert agent_builder._timeout_policy == prod_builder._timeout_policy

    def test_same_rate_limit_policy(self):
        agent_builder = LLMBuilder().for_agent_runtime()
        prod_builder = LLMBuilder().profile("production")
        assert agent_builder._rate_limit_policy == prod_builder._rate_limit_policy

    def test_same_breaker_policy(self):
        agent_builder = LLMBuilder().for_agent_runtime()
        prod_builder = LLMBuilder().profile("production")
        assert agent_builder._breaker_policy == prod_builder._breaker_policy

    def test_same_hedging_policy(self):
        agent_builder = LLMBuilder().for_agent_runtime()
        prod_builder = LLMBuilder().profile("production")
        assert agent_builder._hedging_policy == prod_builder._hedging_policy

    def test_same_cache_policy(self):
        agent_builder = LLMBuilder().for_agent_runtime()
        prod_builder = LLMBuilder().profile("production")
        assert agent_builder._cache_policy == prod_builder._cache_policy

    def test_same_coalescing_policy(self):
        agent_builder = LLMBuilder().for_agent_runtime()
        prod_builder = LLMBuilder().profile("production")
        assert agent_builder._coalescing_policy == prod_builder._coalescing_policy


class TestLLMBuilderFluentAPI:
    """Every builder method returns the same builder instance for chaining."""

    def test_provider_returns_self(self):
        b = LLMBuilder()
        result = b.provider("openai")
        assert result is b

    def test_model_returns_self(self):
        b = LLMBuilder()
        result = b.model("gpt-4")
        assert result is b

    def test_settings_returns_self(self):
        b = LLMBuilder()
        result = b.settings(LLMSettings())
        assert result is b

    def test_profile_returns_self(self):
        b = LLMBuilder()
        result = b.profile("development")
        assert result is b

    def test_for_agent_runtime_returns_self(self):
        b = LLMBuilder()
        result = b.for_agent_runtime()
        assert result is b

    def test_with_provider_settings_returns_self(self):
        b = LLMBuilder()
        result = b.with_provider_settings("openai", {"api_key": "test"})
        assert result is b

    def test_with_middlewares_returns_self(self):
        b = LLMBuilder()
        result = b.with_middlewares(MiddlewareStack())
        assert result is b

    def test_with_observers_returns_self(self):
        b = LLMBuilder()
        result = b.with_observers([])
        assert result is b

    def test_with_cache_returns_self(self):
        b = LLMBuilder()
        result = b.with_cache("inmemory")
        assert result is b

    def test_with_router_returns_self(self):
        b = LLMBuilder()
        result = b.with_router("round_robin")
        assert result is b

    def test_full_chain_returns_same_instance(self):
        b = LLMBuilder()
        result = (
            b.provider("openai")
            .model("gpt-4")
            .profile("production")
            .with_provider_settings("openai", {"api_key": "sk-test"})
            .with_middlewares(MiddlewareStack())
            .with_observers([])
            .with_cache("inmemory")
            .with_router("round_robin")
        )
        assert result is b


class TestLLMBuilderWithProviderSettings:
    """`.with_provider_settings()` stores provider-specific settings keyed by lowercase provider id."""

    def test_stores_settings(self):
        b = LLMBuilder().with_provider_settings("openai", {"api_key": "test"})
        assert "openai" in b._provider_settings
        assert b._provider_settings["openai"]["api_key"] == "test"

    def test_lowercases_provider_key(self):
        b = LLMBuilder().with_provider_settings("OpenAI", {"api_key": "test"})
        assert "openai" in b._provider_settings

    def test_strips_provider_key(self):
        b = LLMBuilder().with_provider_settings("  OpenAI  ", {"api_key": "test"})
        assert "openai" in b._provider_settings

    def test_multiple_providers(self):
        b = (
            LLMBuilder()
            .with_provider_settings("openai", {"api_key": "sk-openai"})
            .with_provider_settings("anthropic", {"api_key": "sk-anthropic"})
        )
        assert len(b._provider_settings) == 2
        assert b._provider_settings["openai"]["api_key"] == "sk-openai"
        assert b._provider_settings["anthropic"]["api_key"] == "sk-anthropic"

    def test_overwrites_same_provider(self):
        b = (
            LLMBuilder()
            .with_provider_settings("openai", {"api_key": "old"})
            .with_provider_settings("openai", {"api_key": "new"})
        )
        assert b._provider_settings["openai"]["api_key"] == "new"


class TestLLMBuilderWithMiddlewares:
    """`.with_middlewares()` stores the middleware stack."""

    def test_stores_middleware_stack(self):
        mw = MiddlewareStack()
        b = LLMBuilder().with_middlewares(mw)
        assert b._middlewares is mw


class TestLLMBuilderWithObservers:
    """`.with_observers()` stores a copy of the observer list."""

    def test_stores_observers(self):
        observers = [lambda e: None]
        b = LLMBuilder().with_observers(observers)
        assert b._observers is not None
        assert len(b._observers) == 1

    def test_creates_copy_of_list(self):
        observers = [lambda e: None]
        b = LLMBuilder().with_observers(observers)
        assert b._observers is not observers  # should be a new list

    def test_empty_observers(self):
        b = LLMBuilder().with_observers([])
        assert b._observers == []


class TestLLMBuilderWithCache:
    """`.with_cache()` stores the cache backend."""

    def test_stores_cache_backend(self):
        b = LLMBuilder().with_cache("inmemory")
        assert b._cache_backend == "inmemory"


class TestLLMBuilderWithRouter:
    """`.with_router()` stores the router."""

    def test_stores_router(self):
        b = LLMBuilder().with_router("round_robin")
        assert b._router == "round_robin"


# ============================== PROFILES ==============================


class TestProfilesKeys:
    """PROFILES dictionary should have all expected profile names."""

    def test_has_development(self):
        assert "development" in PROFILES

    def test_has_production(self):
        assert "production" in PROFILES

    def test_has_high_throughput(self):
        assert "high_throughput" in PROFILES

    def test_has_low_latency(self):
        assert "low_latency" in PROFILES

    def test_exact_key_count(self):
        assert len(PROFILES) == 4

    def test_all_expected_keys(self):
        expected = {"development", "production", "high_throughput", "low_latency"}
        assert set(PROFILES.keys()) == expected


class TestProfilesStructure:
    """Every LLMProfile must have all 7 required policy attributes with correct types."""

    _REQUIRED_ATTRS = ("retry", "timeout", "rate_limit", "breaker", "hedging", "cache", "coalescing")
    _EXPECTED_TYPES = {
        "retry": RetryPolicy,
        "timeout": TimeoutPolicy,
        "rate_limit": RateLimitPolicy,
        "breaker": CircuitBreakerPolicy,
        "hedging": HedgingPolicy,
        "cache": CachePolicy,
        "coalescing": CoalescingPolicy,
    }

    @pytest.mark.parametrize("profile_name", ["development", "production", "high_throughput", "low_latency"])
    def test_has_all_required_attrs(self, profile_name):
        profile = PROFILES[profile_name]
        for attr in self._REQUIRED_ATTRS:
            assert hasattr(profile, attr), f"Profile '{profile_name}' missing attr '{attr}'"

    @pytest.mark.parametrize("profile_name", ["development", "production", "high_throughput", "low_latency"])
    def test_correct_policy_types(self, profile_name):
        profile = PROFILES[profile_name]
        for attr, expected_type in self._EXPECTED_TYPES.items():
            assert isinstance(getattr(profile, attr), expected_type), (
                f"Profile '{profile_name}' attr '{attr}' expected {expected_type.__name__}, "
                f"got {type(getattr(profile, attr)).__name__}"
            )

    @pytest.mark.parametrize("profile_name", ["development", "production", "high_throughput", "low_latency"])
    def test_has_name(self, profile_name):
        profile = PROFILES[profile_name]
        assert profile.name == profile_name


class TestProductionProfileValues:
    """Verify specific values in the production profile."""

    def test_retry_max_retries(self):
        assert PROFILES["production"].retry.max_retries == 3

    def test_retry_backoff_base(self):
        assert PROFILES["production"].retry.backoff_base_s == 0.5

    def test_retry_backoff_jitter(self):
        assert PROFILES["production"].retry.backoff_jitter_s == 0.15

    def test_timeout_request(self):
        assert PROFILES["production"].timeout.request_timeout_s == 30.0

    def test_timeout_stream_idle(self):
        assert PROFILES["production"].timeout.stream_idle_timeout_s == 45.0

    def test_rate_limit_rps(self):
        assert PROFILES["production"].rate_limit.requests_per_second == 20.0

    def test_rate_limit_burst(self):
        assert PROFILES["production"].rate_limit.burst == 40

    def test_breaker_failure_threshold(self):
        assert PROFILES["production"].breaker.failure_threshold == 5

    def test_breaker_cooldown(self):
        assert PROFILES["production"].breaker.cooldown_s == 30.0

    def test_breaker_half_open_max(self):
        assert PROFILES["production"].breaker.half_open_max_calls == 1

    def test_hedging_disabled(self):
        assert PROFILES["production"].hedging.enabled is False

    def test_cache_disabled(self):
        assert PROFILES["production"].cache.enabled is False

    def test_cache_ttl(self):
        assert PROFILES["production"].cache.ttl_s == 30.0

    def test_coalescing_enabled(self):
        assert PROFILES["production"].coalescing.enabled is True


class TestLowLatencyProfileValues:
    """Low latency profile has hedging enabled and aggressive timeouts."""

    def test_hedging_enabled(self):
        assert PROFILES["low_latency"].hedging.enabled is True

    def test_hedging_delay(self):
        assert PROFILES["low_latency"].hedging.delay_s == 0.08

    def test_request_timeout(self):
        assert PROFILES["low_latency"].timeout.request_timeout_s == 10.0

    def test_stream_idle_timeout(self):
        assert PROFILES["low_latency"].timeout.stream_idle_timeout_s == 20.0

    def test_cache_enabled(self):
        assert PROFILES["low_latency"].cache.enabled is True

    def test_breaker_failure_threshold(self):
        assert PROFILES["low_latency"].breaker.failure_threshold == 3

    def test_retry_max_retries(self):
        assert PROFILES["low_latency"].retry.max_retries == 1


class TestHighThroughputProfileValues:
    """High throughput profile has high RPS and cache enabled."""

    def test_rate_limit_rps(self):
        assert PROFILES["high_throughput"].rate_limit.requests_per_second == 120.0

    def test_rate_limit_burst(self):
        assert PROFILES["high_throughput"].rate_limit.burst == 200

    def test_cache_enabled(self):
        assert PROFILES["high_throughput"].cache.enabled is True

    def test_cache_ttl(self):
        assert PROFILES["high_throughput"].cache.ttl_s == 20.0

    def test_breaker_failure_threshold(self):
        assert PROFILES["high_throughput"].breaker.failure_threshold == 10


class TestDevelopmentProfileValues:
    """Development profile has relaxed settings."""

    def test_retry_max_retries(self):
        assert PROFILES["development"].retry.max_retries == 1

    def test_rate_limit_rps(self):
        assert PROFILES["development"].rate_limit.requests_per_second == 50.0

    def test_rate_limit_burst(self):
        assert PROFILES["development"].rate_limit.burst == 100

    def test_breaker_failure_threshold(self):
        assert PROFILES["development"].breaker.failure_threshold == 8

    def test_hedging_disabled(self):
        assert PROFILES["development"].hedging.enabled is False

    def test_cache_disabled(self):
        assert PROFILES["development"].cache.enabled is False

    def test_coalescing_enabled(self):
        assert PROFILES["development"].coalescing.enabled is True


# ============================== Structured Output ==============================


class TestJsonSystemPrompt:
    """json_system_prompt() generates a JSON-conformant system prompt."""

    def test_includes_schema_json(self):
        prompt = json_system_prompt(_SampleModel)
        schema = _SampleModel.model_json_schema()
        schema_json = json.dumps(schema, indent=2, ensure_ascii=True)
        assert schema_json in prompt

    def test_includes_json_instructions(self):
        prompt = json_system_prompt(_SampleModel)
        assert "valid JSON" in prompt

    def test_includes_pydantic_schema_reference(self):
        prompt = json_system_prompt(_SampleModel)
        assert "Pydantic schema" in prompt

    def test_includes_no_markdown_instruction(self):
        prompt = json_system_prompt(_SampleModel)
        assert "markdown" in prompt.lower()

    def test_includes_required_fields_rule(self):
        prompt = json_system_prompt(_SampleModel)
        assert "required fields" in prompt.lower()

    def test_includes_double_quotes_rule(self):
        prompt = json_system_prompt(_SampleModel)
        assert "double quotes" in prompt

    def test_output_is_string(self):
        prompt = json_system_prompt(_SampleModel)
        assert isinstance(prompt, str)

    def test_schema_contains_model_properties(self):
        prompt = json_system_prompt(_SampleModel)
        assert '"name"' in prompt
        assert '"value"' in prompt


class TestParseAndValidateJson:
    """parse_and_validate_json() extracts, parses, and validates JSON."""

    def test_valid_json_string(self):
        text = '{"name": "alice", "value": 42}'
        result = parse_and_validate_json(text, _SampleModel)
        assert isinstance(result, _SampleModel)
        assert result.name == "alice"
        assert result.value == 42

    def test_json_with_surrounding_whitespace(self):
        text = '  \n  {"name": "bob", "value": 7}  \n  '
        result = parse_and_validate_json(text, _SampleModel)
        assert result.name == "bob"
        assert result.value == 7

    def test_markdown_fenced_json(self):
        text = '```json\n{"name": "carol", "value": 99}\n```'
        result = parse_and_validate_json(text, _SampleModel)
        assert isinstance(result, _SampleModel)
        assert result.name == "carol"
        assert result.value == 99

    def test_markdown_fenced_json_no_language_tag(self):
        text = '```\n{"name": "dave", "value": 5}\n```'
        result = parse_and_validate_json(text, _SampleModel)
        assert result.name == "dave"
        assert result.value == 5

    def test_json_with_surrounding_text(self):
        text = 'Here is the result: {"name": "eve", "value": 10} end'
        result = parse_and_validate_json(text, _SampleModel)
        assert result.name == "eve"
        assert result.value == 10

    def test_invalid_json_raises_error(self):
        text = "this is not json at all"
        with pytest.raises(LLMInvalidResponseError):
            parse_and_validate_json(text, _SampleModel)

    def test_empty_string_raises_error(self):
        with pytest.raises(LLMInvalidResponseError):
            parse_and_validate_json("", _SampleModel)

    def test_wrong_schema_raises_error(self):
        text = '{"name": "alice", "value": 42}'
        with pytest.raises(LLMInvalidResponseError):
            parse_and_validate_json(text, _OtherModel)

    def test_missing_required_field_raises_error(self):
        text = '{"name": "alice"}'
        with pytest.raises(LLMInvalidResponseError):
            parse_and_validate_json(text, _SampleModel)

    def test_wrong_type_raises_error(self):
        text = '{"name": "alice", "value": "not_an_int"}'
        with pytest.raises(LLMInvalidResponseError):
            parse_and_validate_json(text, _SampleModel)

    def test_json_array_raises_error(self):
        """safe_json_loads returns None for non-dict JSON, like arrays."""
        text = '[{"name": "alice", "value": 42}]'
        with pytest.raises(LLMInvalidResponseError):
            parse_and_validate_json(text, _SampleModel)

    def test_error_message_contains_original_text(self):
        text = "garbage content here"
        with pytest.raises(LLMInvalidResponseError, match="garbage content here"):
            parse_and_validate_json(text, _SampleModel)

    def test_validation_error_includes_schema_info(self):
        text = '{"name": "alice"}'  # missing 'value'
        with pytest.raises(LLMInvalidResponseError, match="schema"):
            parse_and_validate_json(text, _SampleModel)


class TestMakeRepairPrompt:
    """make_repair_prompt() builds a follow-up prompt for repairing invalid JSON."""

    def test_includes_schema_json(self):
        prompt = make_repair_prompt("bad output", _SampleModel)
        schema = _SampleModel.model_json_schema()
        schema_json = json.dumps(schema, indent=2, ensure_ascii=True)
        assert schema_json in prompt

    def test_includes_invalid_response_text(self):
        invalid = "this was the broken response"
        prompt = make_repair_prompt(invalid, _SampleModel)
        assert invalid in prompt

    def test_includes_fix_instruction(self):
        prompt = make_repair_prompt("bad", _SampleModel)
        assert "Fix" in prompt or "fix" in prompt

    def test_includes_schema_reference(self):
        prompt = make_repair_prompt("bad", _SampleModel)
        assert "Pydantic schema" in prompt

    def test_includes_no_explanations_instruction(self):
        prompt = make_repair_prompt("bad", _SampleModel)
        assert "no explanations" in prompt.lower() or "no explanation" in prompt.lower()

    def test_output_is_string(self):
        prompt = make_repair_prompt("bad", _SampleModel)
        assert isinstance(prompt, str)

    def test_empty_invalid_response(self):
        prompt = make_repair_prompt("", _SampleModel)
        assert isinstance(prompt, str)
        assert len(prompt) > 0


# ============================== LLMLifecycleEvent ==============================


class TestLLMLifecycleEvent:
    """LLMLifecycleEvent dataclass fields and defaults."""

    def test_required_fields(self):
        event = LLMLifecycleEvent(
            event_type="request_start",
            request_id="req-1",
            provider_id="openai",
        )
        assert event.event_type == "request_start"
        assert event.request_id == "req-1"
        assert event.provider_id == "openai"

    def test_default_model_is_none(self):
        event = LLMLifecycleEvent(
            event_type="request_start",
            request_id="req-1",
            provider_id="openai",
        )
        assert event.model is None

    def test_default_attempt_is_none(self):
        event = LLMLifecycleEvent(
            event_type="retry",
            request_id="req-1",
            provider_id="openai",
        )
        assert event.attempt is None

    def test_default_latency_ms_is_none(self):
        event = LLMLifecycleEvent(
            event_type="request_success",
            request_id="req-1",
            provider_id="openai",
        )
        assert event.latency_ms is None

    def test_default_usage_is_none(self):
        event = LLMLifecycleEvent(
            event_type="request_success",
            request_id="req-1",
            provider_id="openai",
        )
        assert event.usage is None

    def test_default_error_class_is_none(self):
        event = LLMLifecycleEvent(
            event_type="request_error",
            request_id="req-1",
            provider_id="openai",
        )
        assert event.error_class is None

    def test_default_error_message_is_none(self):
        event = LLMLifecycleEvent(
            event_type="request_error",
            request_id="req-1",
            provider_id="openai",
        )
        assert event.error_message is None

    def test_all_optional_fields_set(self):
        from afk.llms.types import Usage

        usage = Usage(input_tokens=100, output_tokens=50, total_tokens=150)
        event = LLMLifecycleEvent(
            event_type="request_success",
            request_id="req-1",
            provider_id="openai",
            model="gpt-4",
            attempt=2,
            latency_ms=123.4,
            usage=usage,
            error_class="TimeoutError",
            error_message="request timed out",
        )
        assert event.model == "gpt-4"
        assert event.attempt == 2
        assert event.latency_ms == 123.4
        assert event.usage is usage
        assert event.error_class == "TimeoutError"
        assert event.error_message == "request timed out"

    def test_frozen_dataclass(self):
        event = LLMLifecycleEvent(
            event_type="request_start",
            request_id="req-1",
            provider_id="openai",
        )
        with pytest.raises(AttributeError):
            event.event_type = "retry"  # type: ignore[misc]

    def test_has_slots(self):
        assert hasattr(LLMLifecycleEvent, "__slots__")


class TestLLMLifecycleEventType:
    """LLMLifecycleEventType should accept all documented literal values."""

    _VALID_TYPES = [
        "request_start",
        "retry",
        "request_success",
        "request_error",
        "stream_event",
        "cancel",
        "interrupt",
    ]

    @pytest.mark.parametrize("event_type", _VALID_TYPES)
    def test_valid_event_type(self, event_type):
        event = LLMLifecycleEvent(
            event_type=event_type,
            request_id="req-1",
            provider_id="openai",
        )
        assert event.event_type == event_type

    def test_all_expected_event_types_count(self):
        """Verify the count of known event types matches expectations."""
        assert len(self._VALID_TYPES) == 7


# ============================== MiddlewareStack ==============================


class TestMiddlewareStackDefaultConstructor:
    """Default MiddlewareStack constructor sets empty lists."""

    def test_chat_is_empty_list(self):
        mw = MiddlewareStack()
        assert mw.chat == []

    def test_embed_is_empty_list(self):
        mw = MiddlewareStack()
        assert mw.embed == []

    def test_stream_is_empty_list(self):
        mw = MiddlewareStack()
        assert mw.stream == []

    def test_chat_is_list_type(self):
        mw = MiddlewareStack()
        assert isinstance(mw.chat, list)

    def test_embed_is_list_type(self):
        mw = MiddlewareStack()
        assert isinstance(mw.embed, list)

    def test_stream_is_list_type(self):
        mw = MiddlewareStack()
        assert isinstance(mw.stream, list)


class TestMiddlewareStackWithArguments:
    """MiddlewareStack with provided lists stores them."""

    def test_chat_list_provided(self):
        sentinel = object()
        mw = MiddlewareStack(chat=[sentinel])
        assert mw.chat == [sentinel]

    def test_embed_list_provided(self):
        sentinel = object()
        mw = MiddlewareStack(embed=[sentinel])
        assert mw.embed == [sentinel]

    def test_stream_list_provided(self):
        sentinel = object()
        mw = MiddlewareStack(stream=[sentinel])
        assert mw.stream == [sentinel]

    def test_none_chat_becomes_empty_list(self):
        mw = MiddlewareStack(chat=None)
        assert mw.chat == []

    def test_none_embed_becomes_empty_list(self):
        mw = MiddlewareStack(embed=None)
        assert mw.embed == []

    def test_none_stream_becomes_empty_list(self):
        mw = MiddlewareStack(stream=None)
        assert mw.stream == []

    def test_mixed_arguments(self):
        chat_mw = [object()]
        embed_mw = [object(), object()]
        mw = MiddlewareStack(chat=chat_mw, embed=embed_mw)
        assert len(mw.chat) == 1
        assert len(mw.embed) == 2
        assert mw.stream == []


class TestMiddlewareStackFieldAccess:
    """Direct field access on MiddlewareStack."""

    def test_set_chat_field(self):
        mw = MiddlewareStack()
        new_list = [object()]
        mw.chat = new_list
        assert mw.chat is new_list

    def test_set_embed_field(self):
        mw = MiddlewareStack()
        new_list = [object()]
        mw.embed = new_list
        assert mw.embed is new_list

    def test_set_stream_field(self):
        mw = MiddlewareStack()
        new_list = [object()]
        mw.stream = new_list
        assert mw.stream is new_list
