"""
Comprehensive tests for LLM settings, config, and runtime policy defaults:
  - LLMSettings defaults and from_env()
  - LLMConfig defaults and from_env()
  - All runtime policy dataclass defaults
"""

from __future__ import annotations

import os

import pytest

from afk.llms.config import LLMConfig
from afk.llms.runtime.contracts import (
    CachePolicy,
    CircuitBreakerPolicy,
    CoalescingPolicy,
    HedgingPolicy,
    RateLimitPolicy,
    RetryPolicy,
    RoutePolicy,
    TimeoutPolicy,
)
from afk.llms.settings import LLMSettings


# ============================== Helpers ==============================

# All AFK_LLM_* env vars that LLMSettings.from_env() and LLMConfig.from_env() read.
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


# ============================== LLMSettings ==============================


class TestLLMSettingsDefaults:
    """Verify that constructing LLMSettings() with no arguments gives the documented defaults."""

    def test_default_provider(self):
        s = LLMSettings()
        assert s.default_provider == "litellm"

    def test_default_model(self):
        s = LLMSettings()
        assert s.default_model == "gpt-4.1-mini"

    def test_timeout_s(self):
        s = LLMSettings()
        assert s.timeout_s == 30.0

    def test_max_retries(self):
        s = LLMSettings()
        assert s.max_retries == 3

    def test_backoff_base_s(self):
        s = LLMSettings()
        assert s.backoff_base_s == 0.5

    def test_backoff_jitter_s(self):
        s = LLMSettings()
        assert s.backoff_jitter_s == 0.15

    def test_json_max_retries(self):
        s = LLMSettings()
        assert s.json_max_retries == 2

    def test_max_input_chars(self):
        s = LLMSettings()
        assert s.max_input_chars == 200_000

    def test_stream_idle_timeout_s(self):
        s = LLMSettings()
        assert s.stream_idle_timeout_s == 45.0

    def test_embedding_model_is_none(self):
        s = LLMSettings()
        assert s.embedding_model is None

    def test_api_base_url_is_none(self):
        s = LLMSettings()
        assert s.api_base_url is None

    def test_api_key_is_none(self):
        s = LLMSettings()
        assert s.api_key is None

    def test_frozen(self):
        s = LLMSettings()
        with pytest.raises(AttributeError):
            s.default_model = "other"  # type: ignore[misc]


class TestLLMSettingsFromEnvDefaults:
    """from_env() with a clean environment should produce the same defaults."""

    def test_from_env_defaults_match_constructor_defaults(self):
        s = LLMSettings.from_env()
        assert s.default_provider == "litellm"
        assert s.default_model == "gpt-4.1-mini"
        assert s.timeout_s == 30.0
        assert s.max_retries == 3
        assert s.backoff_base_s == 0.5
        assert s.backoff_jitter_s == 0.15
        assert s.json_max_retries == 2
        assert s.max_input_chars == 200_000
        assert s.stream_idle_timeout_s == 45.0
        assert s.embedding_model is None
        assert s.api_base_url is None
        assert s.api_key is None


class TestLLMSettingsFromEnvOverrides:
    """from_env() should pick up every AFK_LLM_* environment variable."""

    def test_override_provider(self):
        os.environ["AFK_LLM_PROVIDER"] = "openai"
        s = LLMSettings.from_env()
        assert s.default_provider == "openai"

    def test_override_model(self):
        os.environ["AFK_LLM_MODEL"] = "claude-sonnet-4-20250514"
        s = LLMSettings.from_env()
        assert s.default_model == "claude-sonnet-4-20250514"

    def test_override_embed_model(self):
        os.environ["AFK_EMBED_MODEL"] = "text-embedding-3-small"
        s = LLMSettings.from_env()
        assert s.embedding_model == "text-embedding-3-small"

    def test_override_api_base_url(self):
        os.environ["AFK_LLM_API_BASE_URL"] = "https://my-proxy.example.com"
        s = LLMSettings.from_env()
        assert s.api_base_url == "https://my-proxy.example.com"

    def test_override_api_key(self):
        os.environ["AFK_LLM_API_KEY"] = "sk-test-key"
        s = LLMSettings.from_env()
        assert s.api_key == "sk-test-key"

    def test_override_timeout_s(self):
        os.environ["AFK_LLM_TIMEOUT_S"] = "60.5"
        s = LLMSettings.from_env()
        assert s.timeout_s == 60.5

    def test_override_max_retries(self):
        os.environ["AFK_LLM_MAX_RETRIES"] = "5"
        s = LLMSettings.from_env()
        assert s.max_retries == 5

    def test_override_backoff_base_s(self):
        os.environ["AFK_LLM_BACKOFF_BASE_S"] = "1.0"
        s = LLMSettings.from_env()
        assert s.backoff_base_s == 1.0

    def test_override_backoff_jitter_s(self):
        os.environ["AFK_LLM_BACKOFF_JITTER_S"] = "0.3"
        s = LLMSettings.from_env()
        assert s.backoff_jitter_s == 0.3

    def test_override_json_max_retries(self):
        os.environ["AFK_LLM_JSON_MAX_RETRIES"] = "4"
        s = LLMSettings.from_env()
        assert s.json_max_retries == 4

    def test_override_max_input_chars(self):
        os.environ["AFK_LLM_MAX_INPUT_CHARS"] = "500000"
        s = LLMSettings.from_env()
        assert s.max_input_chars == 500_000

    def test_override_stream_idle_timeout_s(self):
        os.environ["AFK_LLM_STREAM_IDLE_TIMEOUT_S"] = "90"
        s = LLMSettings.from_env()
        assert s.stream_idle_timeout_s == 90.0

    def test_override_all_vars_at_once(self):
        os.environ["AFK_LLM_PROVIDER"] = "anthropic"
        os.environ["AFK_LLM_MODEL"] = "claude-opus-4-20250514"
        os.environ["AFK_EMBED_MODEL"] = "embed-v2"
        os.environ["AFK_LLM_API_BASE_URL"] = "http://localhost:8080"
        os.environ["AFK_LLM_API_KEY"] = "test-key-123"
        os.environ["AFK_LLM_TIMEOUT_S"] = "120"
        os.environ["AFK_LLM_MAX_RETRIES"] = "10"
        os.environ["AFK_LLM_BACKOFF_BASE_S"] = "2.0"
        os.environ["AFK_LLM_BACKOFF_JITTER_S"] = "0.5"
        os.environ["AFK_LLM_JSON_MAX_RETRIES"] = "5"
        os.environ["AFK_LLM_MAX_INPUT_CHARS"] = "1000000"
        os.environ["AFK_LLM_STREAM_IDLE_TIMEOUT_S"] = "180"

        s = LLMSettings.from_env()
        assert s.default_provider == "anthropic"
        assert s.default_model == "claude-opus-4-20250514"
        assert s.embedding_model == "embed-v2"
        assert s.api_base_url == "http://localhost:8080"
        assert s.api_key == "test-key-123"
        assert s.timeout_s == 120.0
        assert s.max_retries == 10
        assert s.backoff_base_s == 2.0
        assert s.backoff_jitter_s == 0.5
        assert s.json_max_retries == 5
        assert s.max_input_chars == 1_000_000
        assert s.stream_idle_timeout_s == 180.0


class TestLLMSettingsFromEnvInvalidValues:
    """from_env() should fall back to defaults when env vars are invalid."""

    def test_invalid_float_timeout_s(self):
        os.environ["AFK_LLM_TIMEOUT_S"] = "not_a_float"
        s = LLMSettings.from_env()
        assert s.timeout_s == LLMSettings().timeout_s

    def test_invalid_float_backoff_base_s(self):
        os.environ["AFK_LLM_BACKOFF_BASE_S"] = "abc"
        s = LLMSettings.from_env()
        assert s.backoff_base_s == LLMSettings().backoff_base_s

    def test_invalid_float_backoff_jitter_s(self):
        os.environ["AFK_LLM_BACKOFF_JITTER_S"] = "xyz"
        s = LLMSettings.from_env()
        assert s.backoff_jitter_s == LLMSettings().backoff_jitter_s

    def test_invalid_float_stream_idle_timeout_s(self):
        os.environ["AFK_LLM_STREAM_IDLE_TIMEOUT_S"] = "nope"
        s = LLMSettings.from_env()
        assert s.stream_idle_timeout_s == LLMSettings().stream_idle_timeout_s

    def test_invalid_int_max_retries(self):
        os.environ["AFK_LLM_MAX_RETRIES"] = "three"
        s = LLMSettings.from_env()
        assert s.max_retries == LLMSettings().max_retries

    def test_invalid_int_json_max_retries(self):
        os.environ["AFK_LLM_JSON_MAX_RETRIES"] = "two"
        s = LLMSettings.from_env()
        assert s.json_max_retries == LLMSettings().json_max_retries

    def test_invalid_int_max_input_chars(self):
        os.environ["AFK_LLM_MAX_INPUT_CHARS"] = "lots"
        s = LLMSettings.from_env()
        assert s.max_input_chars == LLMSettings().max_input_chars


class TestLLMSettingsToLegacyConfig:
    """to_legacy_config() should produce a correctly-mapped LLMConfig."""

    def test_to_legacy_config_maps_all_fields(self):
        s = LLMSettings(
            default_provider="openai",
            default_model="gpt-4o",
            embedding_model="text-embedding-3-large",
            api_base_url="https://api.example.com",
            api_key="sk-secret",
            timeout_s=60.0,
            max_retries=5,
            backoff_base_s=1.0,
            backoff_jitter_s=0.25,
            json_max_retries=4,
            max_input_chars=300_000,
            stream_idle_timeout_s=90.0,
        )
        cfg = s.to_legacy_config()

        assert isinstance(cfg, LLMConfig)
        assert cfg.default_model == "gpt-4o"
        assert cfg.embedding_model == "text-embedding-3-large"
        assert cfg.timeout_s == 60.0
        assert cfg.max_retries == 5
        assert cfg.backoff_base_s == 1.0
        assert cfg.backoff_jitter_s == 0.25
        assert cfg.json_max_retries == 4
        assert cfg.max_input_chars == 300_000
        assert cfg.api_base_url == "https://api.example.com"
        assert cfg.api_key == "sk-secret"

    def test_to_legacy_config_defaults(self):
        s = LLMSettings()
        cfg = s.to_legacy_config()
        assert cfg.default_model == "gpt-4.1-mini"
        assert cfg.embedding_model is None
        assert cfg.timeout_s == 30.0
        assert cfg.max_retries == 3
        assert cfg.backoff_base_s == 0.5
        assert cfg.backoff_jitter_s == 0.15
        assert cfg.json_max_retries == 2
        assert cfg.max_input_chars == 200_000
        assert cfg.api_base_url is None
        assert cfg.api_key is None

    def test_to_legacy_config_does_not_include_provider(self):
        """LLMConfig does not have a provider field; ensure it is not leaked."""
        s = LLMSettings(default_provider="anthropic")
        cfg = s.to_legacy_config()
        assert not hasattr(cfg, "default_provider")

    def test_to_legacy_config_does_not_include_stream_idle_timeout(self):
        """LLMConfig does not have stream_idle_timeout_s; ensure it is not leaked."""
        s = LLMSettings(stream_idle_timeout_s=99.0)
        cfg = s.to_legacy_config()
        assert not hasattr(cfg, "stream_idle_timeout_s")


# ============================== LLMConfig ==============================


class TestLLMConfigFromEnvDefaults:
    """LLMConfig.from_env() with a clean environment produces correct defaults."""

    def test_from_env_defaults(self):
        cfg = LLMConfig.from_env()
        assert cfg.default_model == "gpt-4.1-mini"
        assert cfg.embedding_model is None
        assert cfg.timeout_s == 30.0
        assert cfg.max_retries == 3
        assert cfg.backoff_base_s == 0.5
        assert cfg.backoff_jitter_s == 0.15
        assert cfg.json_max_retries == 2
        assert cfg.max_input_chars == 200_000
        assert cfg.api_base_url is None
        assert cfg.api_key is None


class TestLLMConfigFromEnvOverrides:
    """LLMConfig.from_env() should pick up each relevant env var."""

    def test_override_model(self):
        os.environ["AFK_LLM_MODEL"] = "gpt-4o"
        cfg = LLMConfig.from_env()
        assert cfg.default_model == "gpt-4o"

    def test_override_embed_model(self):
        os.environ["AFK_EMBED_MODEL"] = "text-embedding-3-small"
        cfg = LLMConfig.from_env()
        assert cfg.embedding_model == "text-embedding-3-small"

    def test_override_api_base_url(self):
        os.environ["AFK_LLM_API_BASE_URL"] = "https://proxy.example.com"
        cfg = LLMConfig.from_env()
        assert cfg.api_base_url == "https://proxy.example.com"

    def test_override_api_key(self):
        os.environ["AFK_LLM_API_KEY"] = "sk-test-key"
        cfg = LLMConfig.from_env()
        assert cfg.api_key == "sk-test-key"

    def test_override_timeout_s(self):
        os.environ["AFK_LLM_TIMEOUT_S"] = "90.5"
        cfg = LLMConfig.from_env()
        assert cfg.timeout_s == 90.5

    def test_override_max_retries(self):
        os.environ["AFK_LLM_MAX_RETRIES"] = "7"
        cfg = LLMConfig.from_env()
        assert cfg.max_retries == 7

    def test_override_backoff_base_s(self):
        os.environ["AFK_LLM_BACKOFF_BASE_S"] = "2.5"
        cfg = LLMConfig.from_env()
        assert cfg.backoff_base_s == 2.5

    def test_override_backoff_jitter_s(self):
        os.environ["AFK_LLM_BACKOFF_JITTER_S"] = "0.4"
        cfg = LLMConfig.from_env()
        assert cfg.backoff_jitter_s == 0.4

    def test_override_json_max_retries(self):
        os.environ["AFK_LLM_JSON_MAX_RETRIES"] = "6"
        cfg = LLMConfig.from_env()
        assert cfg.json_max_retries == 6

    def test_override_max_input_chars(self):
        os.environ["AFK_LLM_MAX_INPUT_CHARS"] = "400000"
        cfg = LLMConfig.from_env()
        assert cfg.max_input_chars == 400_000

    def test_override_all_vars_at_once(self):
        os.environ["AFK_LLM_MODEL"] = "claude-sonnet-4-20250514"
        os.environ["AFK_EMBED_MODEL"] = "embed-v3"
        os.environ["AFK_LLM_API_BASE_URL"] = "http://localhost:9090"
        os.environ["AFK_LLM_API_KEY"] = "key-abc"
        os.environ["AFK_LLM_TIMEOUT_S"] = "45"
        os.environ["AFK_LLM_MAX_RETRIES"] = "8"
        os.environ["AFK_LLM_BACKOFF_BASE_S"] = "1.5"
        os.environ["AFK_LLM_BACKOFF_JITTER_S"] = "0.2"
        os.environ["AFK_LLM_JSON_MAX_RETRIES"] = "3"
        os.environ["AFK_LLM_MAX_INPUT_CHARS"] = "750000"

        cfg = LLMConfig.from_env()
        assert cfg.default_model == "claude-sonnet-4-20250514"
        assert cfg.embedding_model == "embed-v3"
        assert cfg.api_base_url == "http://localhost:9090"
        assert cfg.api_key == "key-abc"
        assert cfg.timeout_s == 45.0
        assert cfg.max_retries == 8
        assert cfg.backoff_base_s == 1.5
        assert cfg.backoff_jitter_s == 0.2
        assert cfg.json_max_retries == 3
        assert cfg.max_input_chars == 750_000


class TestLLMConfigFieldMapping:
    """Ensure all expected fields exist on LLMConfig and are the right types."""

    def test_frozen(self):
        cfg = LLMConfig.from_env()
        with pytest.raises(AttributeError):
            cfg.default_model = "other"  # type: ignore[misc]

    def test_all_fields_present(self):
        cfg = LLMConfig.from_env()
        expected_fields = {
            "default_model",
            "embedding_model",
            "timeout_s",
            "max_retries",
            "backoff_base_s",
            "backoff_jitter_s",
            "json_max_retries",
            "max_input_chars",
            "api_base_url",
            "api_key",
        }
        actual_fields = {f.name for f in cfg.__dataclass_fields__.values()}
        assert expected_fields == actual_fields


# ======================= Runtime Contract Policies =======================


class TestRetryPolicyDefaults:
    def test_max_retries(self):
        p = RetryPolicy()
        assert p.max_retries == 3

    def test_backoff_base_s(self):
        p = RetryPolicy()
        assert p.backoff_base_s == 0.5

    def test_backoff_jitter_s(self):
        p = RetryPolicy()
        assert p.backoff_jitter_s == 0.15

    def test_require_idempotency_key(self):
        p = RetryPolicy()
        assert p.require_idempotency_key is True

    def test_frozen(self):
        p = RetryPolicy()
        with pytest.raises(AttributeError):
            p.max_retries = 10  # type: ignore[misc]


class TestTimeoutPolicyDefaults:
    def test_request_timeout_s(self):
        p = TimeoutPolicy()
        assert p.request_timeout_s == 30.0

    def test_stream_idle_timeout_s(self):
        p = TimeoutPolicy()
        assert p.stream_idle_timeout_s == 45.0

    def test_frozen(self):
        p = TimeoutPolicy()
        with pytest.raises(AttributeError):
            p.request_timeout_s = 99.0  # type: ignore[misc]


class TestRateLimitPolicyDefaults:
    def test_requests_per_second(self):
        p = RateLimitPolicy()
        assert p.requests_per_second == 20.0

    def test_burst(self):
        p = RateLimitPolicy()
        assert p.burst == 40

    def test_frozen(self):
        p = RateLimitPolicy()
        with pytest.raises(AttributeError):
            p.burst = 100  # type: ignore[misc]


class TestCircuitBreakerPolicyDefaults:
    def test_failure_threshold(self):
        p = CircuitBreakerPolicy()
        assert p.failure_threshold == 5

    def test_cooldown_s(self):
        p = CircuitBreakerPolicy()
        assert p.cooldown_s == 30.0

    def test_half_open_max_calls(self):
        p = CircuitBreakerPolicy()
        assert p.half_open_max_calls == 1

    def test_frozen(self):
        p = CircuitBreakerPolicy()
        with pytest.raises(AttributeError):
            p.failure_threshold = 50  # type: ignore[misc]


class TestHedgingPolicyDefaults:
    def test_enabled(self):
        p = HedgingPolicy()
        assert p.enabled is False

    def test_delay_s(self):
        p = HedgingPolicy()
        assert p.delay_s == 0.2

    def test_frozen(self):
        p = HedgingPolicy()
        with pytest.raises(AttributeError):
            p.enabled = True  # type: ignore[misc]


class TestCachePolicyDefaults:
    def test_enabled(self):
        p = CachePolicy()
        assert p.enabled is False

    def test_ttl_s(self):
        p = CachePolicy()
        assert p.ttl_s == 30.0

    def test_frozen(self):
        p = CachePolicy()
        with pytest.raises(AttributeError):
            p.ttl_s = 999.0  # type: ignore[misc]


class TestCoalescingPolicyDefaults:
    def test_enabled(self):
        p = CoalescingPolicy()
        assert p.enabled is True

    def test_frozen(self):
        p = CoalescingPolicy()
        with pytest.raises(AttributeError):
            p.enabled = False  # type: ignore[misc]


class TestRoutePolicyDefaults:
    def test_provider_order_is_empty_tuple(self):
        p = RoutePolicy()
        assert p.provider_order == ()
        assert isinstance(p.provider_order, tuple)

    def test_frozen(self):
        p = RoutePolicy()
        with pytest.raises(AttributeError):
            p.provider_order = ("a",)  # type: ignore[misc]

    def test_custom_provider_order(self):
        p = RoutePolicy(provider_order=("openai", "anthropic", "litellm"))
        assert p.provider_order == ("openai", "anthropic", "litellm")
