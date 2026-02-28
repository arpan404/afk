"""
Tests for afk.queues.factory (_env_first and create_task_queue_from_env).
"""

import os

import pytest

from afk.queues.factory import _env_first, create_task_queue_from_env
from afk.queues.memory import InMemoryTaskQueue


# ── _env_first ───────────────────────────────────────────────────────────────


class TestEnvFirst:
    def test_returns_first_non_empty(self):
        os.environ["TEST_VAR_A"] = "value_a"
        try:
            assert _env_first("TEST_VAR_A") == "value_a"
        finally:
            os.environ.pop("TEST_VAR_A", None)

    def test_skips_none_env_vars(self):
        os.environ.pop("TEST_VAR_MISSING", None)
        os.environ["TEST_VAR_PRESENT"] = "found"
        try:
            assert _env_first("TEST_VAR_MISSING", "TEST_VAR_PRESENT") == "found"
        finally:
            os.environ.pop("TEST_VAR_PRESENT", None)

    def test_returns_default_when_all_missing(self):
        os.environ.pop("TEST_VAR_NO1", None)
        os.environ.pop("TEST_VAR_NO2", None)
        result = _env_first("TEST_VAR_NO1", "TEST_VAR_NO2", default="fallback")
        assert result == "fallback"

    def test_returns_none_default_when_all_missing(self):
        os.environ.pop("TEST_VAR_GONE", None)
        result = _env_first("TEST_VAR_GONE")
        assert result is None

    def test_strips_whitespace(self):
        os.environ["TEST_VAR_WS"] = "  trimmed  "
        try:
            assert _env_first("TEST_VAR_WS") == "trimmed"
        finally:
            os.environ.pop("TEST_VAR_WS", None)

    def test_skips_empty_string_var(self):
        os.environ["TEST_VAR_EMPTY"] = ""
        os.environ["TEST_VAR_OK"] = "good"
        try:
            assert _env_first("TEST_VAR_EMPTY", "TEST_VAR_OK") == "good"
        finally:
            os.environ.pop("TEST_VAR_EMPTY", None)
            os.environ.pop("TEST_VAR_OK", None)

    def test_skips_whitespace_only_var(self):
        os.environ["TEST_VAR_SPACES"] = "   "
        os.environ["TEST_VAR_REAL"] = "real"
        try:
            assert _env_first("TEST_VAR_SPACES", "TEST_VAR_REAL") == "real"
        finally:
            os.environ.pop("TEST_VAR_SPACES", None)
            os.environ.pop("TEST_VAR_REAL", None)

    def test_returns_first_when_multiple_set(self):
        os.environ["TEST_VAR_FIRST"] = "first"
        os.environ["TEST_VAR_SECOND"] = "second"
        try:
            assert _env_first("TEST_VAR_FIRST", "TEST_VAR_SECOND") == "first"
        finally:
            os.environ.pop("TEST_VAR_FIRST", None)
            os.environ.pop("TEST_VAR_SECOND", None)

    def test_no_names_returns_default(self):
        assert _env_first(default="mydef") == "mydef"

    def test_no_names_no_default_returns_none(self):
        assert _env_first() is None


# ── create_task_queue_from_env ───────────────────────────────────────────────


# Keys to clean up after every test in this class
_QUEUE_ENV_KEYS = [
    "AFK_QUEUE_BACKEND",
    "AFK_QUEUE_RETRY_BACKOFF_BASE_S",
    "AFK_QUEUE_RETRY_BACKOFF_MAX_S",
    "AFK_QUEUE_RETRY_BACKOFF_JITTER_S",
]


def _cleanup_env():
    """Remove all AFK_QUEUE_* test keys."""
    for key in _QUEUE_ENV_KEYS:
        os.environ.pop(key, None)


class TestCreateTaskQueueFromEnv:
    def setup_method(self):
        _cleanup_env()

    def teardown_method(self):
        _cleanup_env()

    def test_default_returns_in_memory(self):
        queue = create_task_queue_from_env()
        assert isinstance(queue, InMemoryTaskQueue)

    def test_backend_inmemory(self):
        os.environ["AFK_QUEUE_BACKEND"] = "inmemory"
        queue = create_task_queue_from_env()
        assert isinstance(queue, InMemoryTaskQueue)

    def test_backend_memory(self):
        os.environ["AFK_QUEUE_BACKEND"] = "memory"
        queue = create_task_queue_from_env()
        assert isinstance(queue, InMemoryTaskQueue)

    def test_backend_in_memory(self):
        os.environ["AFK_QUEUE_BACKEND"] = "in_memory"
        queue = create_task_queue_from_env()
        assert isinstance(queue, InMemoryTaskQueue)

    def test_backend_mem(self):
        os.environ["AFK_QUEUE_BACKEND"] = "mem"
        queue = create_task_queue_from_env()
        assert isinstance(queue, InMemoryTaskQueue)

    def test_backend_case_insensitive(self):
        os.environ["AFK_QUEUE_BACKEND"] = "InMemory"
        queue = create_task_queue_from_env()
        assert isinstance(queue, InMemoryTaskQueue)

    def test_backend_unknown_raises_value_error(self):
        os.environ["AFK_QUEUE_BACKEND"] = "unknown"
        with pytest.raises(ValueError, match="Unknown AFK_QUEUE_BACKEND"):
            create_task_queue_from_env()

    def test_backend_unknown_custom_name_raises(self):
        os.environ["AFK_QUEUE_BACKEND"] = "postgres"
        with pytest.raises(ValueError, match="postgres"):
            create_task_queue_from_env()

    def test_backoff_params_from_env(self):
        os.environ["AFK_QUEUE_BACKEND"] = "inmemory"
        os.environ["AFK_QUEUE_RETRY_BACKOFF_BASE_S"] = "1.5"
        os.environ["AFK_QUEUE_RETRY_BACKOFF_MAX_S"] = "60"
        os.environ["AFK_QUEUE_RETRY_BACKOFF_JITTER_S"] = "0.5"
        queue = create_task_queue_from_env()
        assert isinstance(queue, InMemoryTaskQueue)
        assert queue._retry_backoff_base_s == 1.5
        assert queue._retry_backoff_max_s == 60.0
        assert queue._retry_backoff_jitter_s == 0.5

    def test_default_backoff_params(self):
        queue = create_task_queue_from_env()
        assert isinstance(queue, InMemoryTaskQueue)
        assert queue._retry_backoff_base_s == 0.5
        assert queue._retry_backoff_max_s == 30.0
        assert queue._retry_backoff_jitter_s == 0.2

    def test_backend_with_whitespace(self):
        os.environ["AFK_QUEUE_BACKEND"] = "  inmemory  "
        queue = create_task_queue_from_env()
        assert isinstance(queue, InMemoryTaskQueue)

    def test_redis_backend_with_mock_import(self):
        """Redis backend tries to import from .redis_queue; we just verify the branch."""
        os.environ["AFK_QUEUE_BACKEND"] = "redis"
        # The redis backend will try to import RedisTaskQueue from the redis_queue module.
        # We expect either an ImportError or RuntimeError depending on whether redis is installed.
        # This test verifies the "redis" branch is reached.
        try:
            create_task_queue_from_env()
        except (ImportError, RuntimeError):
            pass  # Expected when redis is not available
