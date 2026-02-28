"""Comprehensive tests for memory utils, vector utilities, factory, and types."""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import contextmanager

import numpy as np
import pytest

from afk.memory.utils import json_dumps, json_loads, new_id, now_ms
from afk.memory.vector import cosine_similarity, format_pgvector
from afk.memory.factory import _env_bool, create_memory_store_from_env
from afk.memory.adapters.in_memory import InMemoryMemoryStore
from afk.memory.adapters.sqlite import SQLiteMemoryStore
from afk.memory.types import LongTermMemory, MemoryEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextmanager
def preserved_env():
    """Save and restore the full environment around a block."""
    snapshot = os.environ.copy()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


# ===========================================================================
# 1. Memory Utils (afk.memory.utils)
# ===========================================================================


class TestNowMs:
    """Tests for now_ms()."""

    def test_returns_int(self):
        result = now_ms()
        assert isinstance(result, int)

    def test_close_to_current_time(self):
        before = int(time.time() * 1000)
        result = now_ms()
        after = int(time.time() * 1000)
        assert before <= result <= after


class TestNewId:
    """Tests for new_id()."""

    def test_default_prefix_mem(self):
        result = new_id()
        assert result.startswith("mem_")
        # The part after "mem_" should be a 32-char hex string (uuid4 hex)
        hex_part = result[len("mem_"):]
        assert re.fullmatch(r"[0-9a-f]{32}", hex_part), f"Unexpected hex part: {hex_part}"

    def test_custom_prefix(self):
        result = new_id("custom")
        assert result.startswith("custom_")
        hex_part = result[len("custom_"):]
        assert re.fullmatch(r"[0-9a-f]{32}", hex_part), f"Unexpected hex part: {hex_part}"

    def test_unique_on_repeated_calls(self):
        ids = {new_id() for _ in range(100)}
        assert len(ids) == 100, "new_id() should produce unique IDs"


class TestJsonDumps:
    """Tests for json_dumps()."""

    def test_compact_json_no_spaces(self):
        result = json_dumps({"a": 1, "b": 2})
        # Compact separators => no spaces around : or ,
        assert " " not in result

    def test_handles_unicode(self):
        result = json_dumps({"emoji": "\u2603", "kanji": "\u6f22\u5b57"})
        # ensure_ascii=False means unicode chars are kept as-is, not escaped
        assert "\u2603" in result
        assert "\u6f22\u5b57" in result
        assert "\\u" not in result

    def test_round_trip(self):
        payload = {"key": [1, 2, None, True], "nested": {"x": "y"}}
        serialized = json_dumps(payload)
        deserialized = json.loads(serialized)
        assert deserialized == payload


class TestJsonLoads:
    """Tests for json_loads()."""

    def test_parses_valid_json(self):
        result = json_loads('{"a":1,"b":[2,3]}')
        assert result == {"a": 1, "b": [2, 3]}

    def test_raises_on_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            json_loads("{invalid json!!!}")


# ===========================================================================
# 2. Vector Utilities (afk.memory.vector)
# ===========================================================================


class TestCosineSimilarity:
    """Tests for cosine_similarity()."""

    def test_identical_nonzero_vectors(self):
        result = cosine_similarity([3.0, 4.0], [3.0, 4.0])
        assert result == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        result = cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_opposite_vectors(self):
        result = cosine_similarity([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0])
        assert result == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        result = cosine_similarity([0.0, 0.0], [1.0, 2.0])
        assert result == pytest.approx(0.0)

    def test_both_zero_vectors_returns_zero(self):
        result = cosine_similarity([0.0, 0.0], [0.0, 0.0])
        assert result == pytest.approx(0.0)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="Embedding dim mismatch"):
            cosine_similarity([1.0, 2.0], [1.0])

    def test_non_1d_input_raises(self):
        with pytest.raises(ValueError, match="1D vectors"):
            cosine_similarity([[1.0, 2.0]], [1.0, 2.0])  # type: ignore[arg-type]

    def test_non_1d_second_arg_raises(self):
        with pytest.raises(ValueError, match="1D vectors"):
            cosine_similarity([1.0, 2.0], [[1.0, 2.0]])  # type: ignore[arg-type]

    def test_works_with_plain_lists(self):
        # Ensure it works even when inputs are plain Python lists (not numpy arrays)
        result = cosine_similarity([1, 0, 0], [1, 0, 0])
        assert result == pytest.approx(1.0)

    def test_works_with_numpy_arrays(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        result = cosine_similarity(a, b)
        assert result == pytest.approx(0.0, abs=1e-9)


class TestFormatPgvector:
    """Tests for format_pgvector()."""

    def test_bracket_delimited_comma_separated(self):
        result = format_pgvector([1.0, 2.5, 3.125])
        assert result == "[1,2.5,3.125]"

    def test_single_element(self):
        result = format_pgvector([42.0])
        assert result == "[42]"

    def test_empty_list(self):
        result = format_pgvector([])
        assert result == "[]"

    def test_integer_inputs(self):
        result = format_pgvector([1, 2, 3])
        assert result == "[1,2,3]"

    def test_negative_values(self):
        result = format_pgvector([-1.5, 0.0, 1.5])
        assert result == "[-1.5,0,1.5]"


# ===========================================================================
# 3. Memory Factory (afk.memory.factory)
# ===========================================================================


class TestEnvBool:
    """Tests for _env_bool()."""

    @pytest.mark.parametrize("value", ["1", "true", "yes", "y", "on",
                                       "TRUE", "True", "YES", "Yes", "Y", "ON", "On"])
    def test_truthy_values(self, value):
        with preserved_env():
            os.environ["TEST_BOOL_FLAG"] = value
            assert _env_bool("TEST_BOOL_FLAG") is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "n", "off", "", "anything", "nope"])
    def test_falsy_values(self, value):
        with preserved_env():
            os.environ["TEST_BOOL_FLAG"] = value
            assert _env_bool("TEST_BOOL_FLAG") is False

    def test_default_when_not_set(self):
        with preserved_env():
            os.environ.pop("TEST_BOOL_FLAG_MISSING", None)
            assert _env_bool("TEST_BOOL_FLAG_MISSING", default=False) is False
            assert _env_bool("TEST_BOOL_FLAG_MISSING", default=True) is True

    def test_whitespace_stripped(self):
        with preserved_env():
            os.environ["TEST_BOOL_FLAG"] = "  true  "
            assert _env_bool("TEST_BOOL_FLAG") is True


class TestCreateMemoryStoreFromEnv:
    """Tests for create_memory_store_from_env()."""

    @pytest.mark.parametrize("backend", ["in_memory", "inmemory", "mem", "memory"])
    def test_in_memory_backend(self, backend):
        with preserved_env():
            os.environ["AFK_MEMORY_BACKEND"] = backend
            store = create_memory_store_from_env()
            assert isinstance(store, InMemoryMemoryStore)

    @pytest.mark.parametrize("backend", ["sqlite", "sqlite3"])
    def test_sqlite_backend(self, tmp_path, backend):
        with preserved_env():
            db_path = str(tmp_path / "test_memory.sqlite3")
            os.environ["AFK_MEMORY_BACKEND"] = backend
            os.environ["AFK_SQLITE_PATH"] = db_path
            store = create_memory_store_from_env()
            assert isinstance(store, SQLiteMemoryStore)
            assert store.path == db_path

    def test_sqlite_backend_default_path(self):
        with preserved_env():
            os.environ["AFK_MEMORY_BACKEND"] = "sqlite"
            os.environ.pop("AFK_SQLITE_PATH", None)
            store = create_memory_store_from_env()
            assert isinstance(store, SQLiteMemoryStore)
            assert store.path == "afk_memory.sqlite3"

    def test_unknown_backend_raises(self):
        with preserved_env():
            os.environ["AFK_MEMORY_BACKEND"] = "unknown_backend_xyz"
            with pytest.raises(ValueError, match="Unknown AFK_MEMORY_BACKEND"):
                create_memory_store_from_env()

    def test_case_insensitive_backend(self):
        with preserved_env():
            os.environ["AFK_MEMORY_BACKEND"] = "IN_MEMORY"
            store = create_memory_store_from_env()
            assert isinstance(store, InMemoryMemoryStore)

    def test_whitespace_in_backend(self):
        with preserved_env():
            os.environ["AFK_MEMORY_BACKEND"] = "  sqlite  "
            os.environ["AFK_SQLITE_PATH"] = ":memory:"
            store = create_memory_store_from_env()
            assert isinstance(store, SQLiteMemoryStore)


# ===========================================================================
# 4. Memory Types - Edge Cases (afk.memory.types)
# ===========================================================================


class TestMemoryEvent:
    """Tests for MemoryEvent dataclass."""

    def test_create_with_all_required_fields(self):
        event = MemoryEvent(
            id="evt_001",
            thread_id="thread_abc",
            user_id="user_123",
            type="message",
            timestamp=1700000000000,
            payload={"content": "hello"},
        )
        assert event.id == "evt_001"
        assert event.thread_id == "thread_abc"
        assert event.user_id == "user_123"
        assert event.type == "message"
        assert event.timestamp == 1700000000000
        assert event.payload == {"content": "hello"}
        assert event.tags == []  # default

    def test_create_with_optional_tags(self):
        event = MemoryEvent(
            id="evt_002",
            thread_id="thread_xyz",
            user_id=None,
            type="tool_call",
            timestamp=1700000000000,
            payload={},
            tags=["important", "pinned"],
        )
        assert event.user_id is None
        assert event.tags == ["important", "pinned"]

    def test_frozen_immutability(self):
        event = MemoryEvent(
            id="evt_003",
            thread_id="t",
            user_id=None,
            type="system",
            timestamp=0,
            payload={},
        )
        with pytest.raises(AttributeError):
            event.id = "changed"  # type: ignore[misc]


class TestLongTermMemory:
    """Tests for LongTermMemory dataclass."""

    def test_default_timestamps_close_to_now(self):
        before = int(time.time() * 1000)
        mem = LongTermMemory(
            id="ltm_001",
            user_id="user_1",
            scope="global",
            data={"preference": "dark_mode"},
        )
        after = int(time.time() * 1000)

        assert before <= mem.created_at <= after
        assert before <= mem.updated_at <= after

    def test_explicit_timestamps(self):
        mem = LongTermMemory(
            id="ltm_002",
            user_id=None,
            scope="project:abc",
            data={},
            created_at=1000,
            updated_at=2000,
        )
        assert mem.created_at == 1000
        assert mem.updated_at == 2000

    def test_defaults_for_optional_fields(self):
        mem = LongTermMemory(
            id="ltm_003",
            user_id="u",
            scope="global",
            data={"key": "value"},
        )
        assert mem.text is None
        assert mem.tags == []
        assert mem.metadata == {}

    def test_frozen_immutability(self):
        mem = LongTermMemory(
            id="ltm_004",
            user_id="u",
            scope="global",
            data={},
        )
        with pytest.raises(AttributeError):
            mem.id = "changed"  # type: ignore[misc]
