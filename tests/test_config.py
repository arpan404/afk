"""
Tests for the centralized config module.
"""

from __future__ import annotations

import pytest

from afk.config import (
    MCPServerEnv,
    MemoryEnv,
    RunnerEnv,
    Settings,
    EnvVarField,
    _bool,
    _csv_list,
    _float,
    _int,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeSettings(Settings):
    """Minimal Settings subclass for testing the base class."""

    STR_VAL: ClassVar[str] = EnvVarField("FAKE_STR", default="default_str")
    INT_VAL: ClassVar[int] = EnvVarField("FAKE_INT", default=42, parser=_int)
    FLOAT_VAL: ClassVar[float] = EnvVarField("FAKE_FLOAT", default=1.5, parser=_float)
    BOOL_VAL: ClassVar[bool] = EnvVarField("FAKE_BOOL", default=False, parser=_bool)
    LIST_VAL: ClassVar[list[str]] = EnvVarField(
        "FAKE_LIST", default=[], parser=_csv_list
    )
    NONE_STR: ClassVar[str] = EnvVarField("FAKE_NONE", default=None)


# ---------------------------------------------------------------------------
# Parser unit tests
# ---------------------------------------------------------------------------


class TestParsers:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1", True),
            ("true", True),
            ("True", True),
            ("yes", True),
            ("y", True),
            ("on", True),
            ("0", False),
            ("false", False),
            ("no", False),
            ("off", False),
        ],
    )
    def test_bool_parses_valid_values(self, raw, expected):
        assert _bool(raw) == expected

    @pytest.mark.parametrize("raw", ["maybe", "2", "tru", "yeah", ""])
    def test_bool_raises_on_invalid_input(self, raw):
        with pytest.raises(ValueError, match="Invalid boolean"):
            _bool(raw)

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("a,b,c", ["a", "b", "c"]),
            ("a, b , c", ["a", "b", "c"]),
            ("one", ["one"]),
            ("a,b,,c", ["a", "b", "c"]),
            ("", []),
            ("  ", []),
        ],
    )
    def test_csv_list_parses_valid_input(self, raw, expected):
        assert _csv_list(raw) == expected

    def test_int_parses_valid_input(self):
        assert _int("123") == 123
        assert _int("-456") == -456

    def test_int_raises_on_invalid_input(self):
        with pytest.raises(ValueError):
            _int("not_a_number")

    def test_float_parses_valid_input(self):
        assert _float("1.5") == 1.5
        assert _float("-0.25") == -0.25

    def test_float_raises_on_invalid_input(self):
        with pytest.raises(ValueError):
            _float("not_a_float")


# ---------------------------------------------------------------------------
# Settings.from_env() tests
# ---------------------------------------------------------------------------


class TestSettingsFromEnv:
    def test_missing_env_vars_use_defaults(self, monkeypatch):
        # Ensure none of the env vars are set
        for var in (
            "FAKE_STR",
            "FAKE_INT",
            "FAKE_FLOAT",
            "FAKE_BOOL",
            "FAKE_LIST",
            "FAKE_NONE",
        ):
            monkeypatch.delenv(var, raising=False)

        settings = FakeSettings.from_env()
        assert settings.STR_VAL == "default_str"
        assert settings.INT_VAL == 42
        assert settings.FLOAT_VAL == 1.5
        assert settings.BOOL_VAL is False
        assert settings.LIST_VAL == []
        assert settings.NONE_STR is None

    def test_env_vars_override_defaults(self, monkeypatch):
        monkeypatch.setenv("FAKE_STR", "from_env")
        monkeypatch.setenv("FAKE_INT", "100")
        monkeypatch.setenv("FAKE_FLOAT", "2.5")
        monkeypatch.setenv("FAKE_BOOL", "true")
        monkeypatch.setenv("FAKE_LIST", "a,b,c")
        monkeypatch.setenv("FAKE_NONE", "non_none")

        settings = FakeSettings.from_env()
        assert settings.STR_VAL == "from_env"
        assert settings.INT_VAL == 100
        assert settings.FLOAT_VAL == 2.5
        assert settings.BOOL_VAL is True
        assert settings.LIST_VAL == ["a", "b", "c"]
        assert settings.NONE_STR == "non_none"

    def test_empty_string_env_var_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("FAKE_STR", "")
        monkeypatch.setenv("FAKE_INT", "")
        monkeypatch.setenv("FAKE_LIST", "")

        settings = FakeSettings.from_env()
        assert settings.STR_VAL == "default_str"
        assert settings.INT_VAL == 42
        assert settings.LIST_VAL == []

    def test_invalid_int_raises(self, monkeypatch):
        monkeypatch.setenv("FAKE_INT", "not_an_int")
        with pytest.raises(ValueError):
            FakeSettings.from_env()

    def test_invalid_float_raises(self, monkeypatch):
        monkeypatch.setenv("FAKE_FLOAT", "not_a_float")
        with pytest.raises(ValueError):
            FakeSettings.from_env()

    def test_invalid_bool_raises(self, monkeypatch):
        monkeypatch.setenv("FAKE_BOOL", "maybe")
        with pytest.raises(ValueError, match="Invalid boolean"):
            FakeSettings.from_env()


# ---------------------------------------------------------------------------
# Concrete env bindings tests
# ---------------------------------------------------------------------------


class TestMCPServerEnv:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("AFK_CORS_ORIGINS", raising=False)
        monkeypatch.delenv("AFK_MCP_NAME", raising=False)
        monkeypatch.delenv("AFK_MCP_PORT", raising=False)

        env = MCPServerEnv.from_env()
        assert env.AFK_CORS_ORIGINS == []
        assert env.AFK_MCP_NAME == "afk-mcp-server"
        assert env.AFK_MCP_PORT == 8000
        assert env.AFK_MCP_ENABLE_SSE is True
        assert env.AFK_MCP_ENABLE_HEALTH is True
        assert env.AFK_MCP_ALLOW_BATCH is True

    def test_cors_origins_from_csv(self, monkeypatch):
        monkeypatch.setenv("AFK_CORS_ORIGINS", "https://example.com, https://app.io")

        env = MCPServerEnv.from_env()
        assert env.AFK_CORS_ORIGINS == ["https://example.com", "https://app.io"]

    def test_cors_origins_empty_string_is_empty_list(self, monkeypatch):
        monkeypatch.setenv("AFK_CORS_ORIGINS", "")
        env = MCPServerEnv.from_env()
        assert env.AFK_CORS_ORIGINS == []

    def test_bool_fields(self, monkeypatch):
        monkeypatch.setenv("AFK_MCP_ENABLE_SSE", "false")
        monkeypatch.setenv("AFK_MCP_ENABLE_HEALTH", "0")
        monkeypatch.setenv("AFK_MCP_ALLOW_BATCH", "no")

        env = MCPServerEnv.from_env()
        assert env.AFK_MCP_ENABLE_SSE is False
        assert env.AFK_MCP_ENABLE_HEALTH is False
        assert env.AFK_MCP_ALLOW_BATCH is False


class TestRunnerEnv:
    def test_empty_allowlist_by_default(self, monkeypatch):
        monkeypatch.delenv("AFK_ALLOWED_COMMANDS", raising=False)
        env = RunnerEnv.from_env()
        assert env.AFK_ALLOWED_COMMANDS == []

    def test_allowlist_from_csv(self, monkeypatch):
        monkeypatch.setenv("AFK_ALLOWED_COMMANDS", "ls, cat, rg, echo")
        env = RunnerEnv.from_env()
        assert env.AFK_ALLOWED_COMMANDS == ["ls", "cat", "rg", "echo"]


class TestMemoryEnv:
    def test_defaults(self, monkeypatch):
        for var in (
            "AFK_MEMORY_BACKEND",
            "AFK_SQLITE_PATH",
            "AFK_REDIS_HOST",
            "AFK_REDIS_PORT",
            "AFK_REDIS_DB",
            "AFK_REDIS_PASSWORD",
            "AFK_REDIS_EVENTS_MAX",
            "AFK_PG_HOST",
            "AFK_PG_PORT",
            "AFK_PG_USER",
            "AFK_PG_PASSWORD",
            "AFK_PG_DB",
            "AFK_PG_SSL",
            "AFK_PG_POOL_MIN",
            "AFK_PG_POOL_MAX",
            "AFK_QUEUE_BACKEND",
            "AFK_QUEUE_RETRY_BACKOFF_BASE_S",
            "AFK_QUEUE_RETRY_BACKOFF_MAX_S",
            "AFK_QUEUE_RETRY_BACKOFF_JITTER_S",
            "AFK_QUEUE_REDIS_PREFIX",
            "AFK_QUEUE_REDIS_HOST",
            "AFK_QUEUE_REDIS_PORT",
            "AFK_QUEUE_REDIS_DB",
            "AFK_QUEUE_REDIS_PASSWORD",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv("AFK_REDIS_URL", raising=False)
        monkeypatch.delenv("AFK_PG_DSN", raising=False)
        monkeypatch.delenv("AFK_VECTOR_DIM", raising=False)
        monkeypatch.delenv("AFK_QUEUE_REDIS_URL", raising=False)

        env = MemoryEnv.from_env()
        assert env.AFK_MEMORY_BACKEND == "sqlite"
        assert env.AFK_SQLITE_PATH == "afk_memory.sqlite3"
        assert env.AFK_REDIS_HOST == "localhost"
        assert env.AFK_REDIS_PORT == 6379
        assert env.AFK_REDIS_DB == 0
        assert env.AFK_REDIS_PASSWORD == ""
        assert env.AFK_REDIS_EVENTS_MAX == 2000
        assert env.AFK_PG_SSL is False
        assert env.AFK_PG_POOL_MIN == 1
        assert env.AFK_PG_POOL_MAX == 10
        assert env.AFK_QUEUE_BACKEND == "inmemory"
        assert env.AFK_QUEUE_RETRY_BACKOFF_BASE_S == 0.5
        assert env.AFK_QUEUE_RETRY_BACKOFF_MAX_S == 30.0
        assert env.AFK_QUEUE_RETRY_BACKOFF_JITTER_S == 0.2
        assert env.AFK_QUEUE_REDIS_PREFIX == "afk:queue"
        assert env.AFK_QUEUE_REDIS_HOST == "localhost"
        assert env.AFK_QUEUE_REDIS_PORT == 6379
        assert env.AFK_QUEUE_REDIS_DB == 0
        assert env.AFK_QUEUE_REDIS_PASSWORD == ""

    def test_redis_port_int(self, monkeypatch):
        monkeypatch.setenv("AFK_REDIS_PORT", "9999")
        env = MemoryEnv.from_env()
        assert env.AFK_REDIS_PORT == 9999

    def test_pg_ssl_bool(self, monkeypatch):
        monkeypatch.setenv("AFK_PG_SSL", "true")
        env = MemoryEnv.from_env()
        assert env.AFK_PG_SSL is True

    def test_backoff_floats(self, monkeypatch):
        monkeypatch.setenv("AFK_QUEUE_RETRY_BACKOFF_BASE_S", "1.0")
        monkeypatch.setenv("AFK_QUEUE_RETRY_BACKOFF_MAX_S", "60.0")
        monkeypatch.setenv("AFK_QUEUE_RETRY_BACKOFF_JITTER_S", "0.5")
        env = MemoryEnv.from_env()
        assert env.AFK_QUEUE_RETRY_BACKOFF_BASE_S == 1.0
        assert env.AFK_QUEUE_RETRY_BACKOFF_MAX_S == 60.0
        assert env.AFK_QUEUE_RETRY_BACKOFF_JITTER_S == 0.5
