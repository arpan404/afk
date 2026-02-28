"""
Comprehensive edge-case tests for afk.memory.lifecycle.

Covers apply_event_retention, apply_state_retention, compact_thread_memory,
and private helpers (_safe_int, _extract_timestamp_ms, _parse_checkpoint_latest_key,
_parse_checkpoint_state_key, _parse_effect_key).
"""

from __future__ import annotations

import asyncio

from afk.memory.adapters.in_memory import InMemoryMemoryStore
from afk.memory.lifecycle import (
    MemoryCompactionResult,
    RetentionPolicy,
    StateRetentionPolicy,
    _extract_timestamp_ms,
    _parse_checkpoint_latest_key,
    _parse_checkpoint_state_key,
    _parse_effect_key,
    _safe_int,
    apply_event_retention,
    apply_state_retention,
    compact_thread_memory,
)
from afk.memory.store import MemoryStore
from afk.memory.types import MemoryEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _event(idx: int, event_type: str = "message", thread_id: str = "t1") -> MemoryEvent:
    """Create a MemoryEvent with a deterministic timestamp equal to *idx*."""
    return MemoryEvent(
        id=f"evt_{idx}",
        thread_id=thread_id,
        user_id="u1",
        type=event_type,  # type: ignore[arg-type]
        timestamp=idx,
        payload={"idx": idx},
    )


def run_async(coro):
    """Run an async coroutine synchronously using asyncio.run()."""
    return asyncio.run(coro)


# ===========================================================================
# 1. apply_event_retention
# ===========================================================================


class TestApplyEventRetentionEmptyEvents:
    """Empty events list returns []."""

    def test_empty_list(self):
        result = apply_event_retention([], policy=RetentionPolicy())
        assert result == []


class TestApplyEventRetentionMaxZero:
    """max_events_per_thread=0 returns []."""

    def test_max_zero(self):
        events = [_event(1), _event(2)]
        result = apply_event_retention(
            events, policy=RetentionPolicy(max_events_per_thread=0)
        )
        assert result == []


class TestApplyEventRetentionBelowMax:
    """Events below max are all retained."""

    def test_all_retained(self):
        events = [_event(i) for i in range(3)]
        policy = RetentionPolicy(max_events_per_thread=10, keep_event_types=[])
        result = apply_event_retention(events, policy=policy)
        assert len(result) == 3
        assert [e.id for e in result] == [e.id for e in events]


class TestApplyEventRetentionAboveMax:
    """Events above max: newest are kept, oldest dropped."""

    def test_newest_kept(self):
        events = [_event(i) for i in range(10)]
        policy = RetentionPolicy(max_events_per_thread=3, keep_event_types=[])
        result = apply_event_retention(events, policy=policy)
        assert len(result) == 3
        assert [e.id for e in result] == ["evt_7", "evt_8", "evt_9"]

    def test_oldest_dropped(self):
        events = [_event(i) for i in range(5)]
        policy = RetentionPolicy(max_events_per_thread=2, keep_event_types=[])
        result = apply_event_retention(events, policy=policy)
        ids = {e.id for e in result}
        assert "evt_0" not in ids
        assert "evt_1" not in ids
        assert "evt_2" not in ids


class TestApplyEventRetentionKeepEventTypes:
    """keep_event_types preserves those events even if oldest."""

    def test_preserved_even_if_oldest(self):
        events = [
            _event(1, "trace"),  # oldest, but protected
            _event(2, "message"),
            _event(3, "message"),
            _event(4, "message"),
            _event(5, "message"),
        ]
        policy = RetentionPolicy(max_events_per_thread=3, keep_event_types=["trace"])
        result = apply_event_retention(events, policy=policy)
        assert len(result) == 3
        result_ids = {e.id for e in result}
        # The trace event must be present
        assert "evt_1" in result_ids
        # The newest non-trace events fill the remaining budget
        assert "evt_5" in result_ids

    def test_multiple_keep_types(self):
        events = [
            _event(1, "trace"),
            _event(2, "system"),
            _event(3, "message"),
            _event(4, "message"),
            _event(5, "message"),
            _event(6, "message"),
        ]
        policy = RetentionPolicy(
            max_events_per_thread=4, keep_event_types=["trace", "system"]
        )
        result = apply_event_retention(events, policy=policy)
        assert len(result) == 4
        result_ids = {e.id for e in result}
        assert "evt_1" in result_ids
        assert "evt_2" in result_ids


class TestApplyEventRetentionPreservedExceedsMax:
    """If preserved events alone exceed max, newest preserved are kept."""

    def test_preserved_trimmed(self):
        # All events are trace (protected), but max is smaller
        events = [_event(i, "trace") for i in range(10)]
        policy = RetentionPolicy(max_events_per_thread=4, keep_event_types=["trace"])
        result = apply_event_retention(events, policy=policy)
        assert len(result) == 4
        # Newest 4 preserved events are kept
        assert [e.id for e in result] == ["evt_6", "evt_7", "evt_8", "evt_9"]


class TestApplyEventRetentionSortedByTimestamp:
    """Output is sorted by timestamp (oldest first)."""

    def test_sorted_output(self):
        # Preserved old event + recent non-preserved events
        events = [
            _event(10, "trace"),
            _event(20, "message"),
            _event(30, "message"),
            _event(40, "message"),
            _event(50, "message"),
        ]
        policy = RetentionPolicy(max_events_per_thread=3, keep_event_types=["trace"])
        result = apply_event_retention(events, policy=policy)
        timestamps = [e.timestamp for e in result]
        assert timestamps == sorted(timestamps)

    def test_sorted_with_interleaved_types(self):
        events = [
            _event(1, "message"),
            _event(2, "trace"),
            _event(3, "message"),
            _event(4, "trace"),
            _event(5, "message"),
        ]
        policy = RetentionPolicy(max_events_per_thread=4, keep_event_types=["trace"])
        result = apply_event_retention(events, policy=policy)
        timestamps = [e.timestamp for e in result]
        assert timestamps == sorted(timestamps)


# ===========================================================================
# 2. apply_state_retention
# ===========================================================================


class TestApplyStateRetentionEmptyState:
    """Empty state returns {}."""

    def test_empty_dict(self):
        assert apply_state_retention({}, policy=StateRetentionPolicy()) == {}


class TestApplyStateRetentionPassthroughKeys:
    """Passthrough keys (non-checkpoint, non-effect) are always kept."""

    def test_custom_keys_preserved(self):
        state = {
            "user:profile": {"name": "Alice"},
            "config:theme": "dark",
        }
        result = apply_state_retention(state, policy=StateRetentionPolicy())
        assert "user:profile" in result
        assert "config:theme" in result
        assert result["user:profile"] == {"name": "Alice"}
        assert result["config:theme"] == "dark"


class TestApplyStateRetentionCheckpointLatest:
    """checkpoint:<run_id>:latest rows are recognized."""

    def test_latest_rows_recognized(self):
        state = {
            "checkpoint:run_A:latest": {"timestamp_ms": 100, "step": 1, "phase": "pre_llm"},
            "checkpoint:run_A:1:pre_llm": {"step": 1, "phase": "pre_llm"},
        }
        result = apply_state_retention(state, policy=StateRetentionPolicy(max_runs=10))
        assert "checkpoint:run_A:latest" in result


class TestApplyStateRetentionMaxRuns:
    """max_runs limits how many runs are retained (most recent kept)."""

    def test_only_recent_runs_kept(self):
        state = {
            "checkpoint:run_old:latest": {"timestamp_ms": 10, "step": 1, "phase": "pre_llm"},
            "checkpoint:run_old:1:pre_llm": {"step": 1, "phase": "pre_llm"},
            "checkpoint:run_mid:latest": {"timestamp_ms": 50, "step": 2, "phase": "pre_llm"},
            "checkpoint:run_mid:2:pre_llm": {"step": 2, "phase": "pre_llm"},
            "checkpoint:run_new:latest": {"timestamp_ms": 100, "step": 3, "phase": "pre_llm"},
            "checkpoint:run_new:3:pre_llm": {"step": 3, "phase": "pre_llm"},
        }
        result = apply_state_retention(
            state, policy=StateRetentionPolicy(max_runs=1)
        )
        assert "checkpoint:run_new:latest" in result
        assert "checkpoint:run_old:latest" not in result
        assert "checkpoint:run_mid:latest" not in result

    def test_two_runs_kept(self):
        state = {
            "checkpoint:run_A:latest": {"timestamp_ms": 10, "step": 1, "phase": "pre_llm"},
            "checkpoint:run_A:1:pre_llm": {"step": 1, "phase": "pre_llm"},
            "checkpoint:run_B:latest": {"timestamp_ms": 50, "step": 2, "phase": "pre_llm"},
            "checkpoint:run_B:2:pre_llm": {"step": 2, "phase": "pre_llm"},
            "checkpoint:run_C:latest": {"timestamp_ms": 100, "step": 3, "phase": "pre_llm"},
            "checkpoint:run_C:3:pre_llm": {"step": 3, "phase": "pre_llm"},
        }
        result = apply_state_retention(
            state, policy=StateRetentionPolicy(max_runs=2)
        )
        assert "checkpoint:run_C:latest" in result
        assert "checkpoint:run_B:latest" in result
        assert "checkpoint:run_A:latest" not in result


class TestApplyStateRetentionAlwaysKeepPhases:
    """always_keep_phases preserves matching checkpoint rows."""

    def test_phases_preserved(self):
        state = {
            "checkpoint:run_1:latest": {
                "timestamp_ms": 100,
                "step": 5,
                "phase": "run_terminal",
            },
            "checkpoint:run_1:5:run_terminal": {"step": 5, "phase": "run_terminal"},
            "checkpoint:run_1:3:pre_llm": {"step": 3, "phase": "pre_llm"},
            "checkpoint:run_1:2:some_custom_phase": {"step": 2, "phase": "some_custom_phase"},
        }
        policy = StateRetentionPolicy(
            max_runs=10,
            always_keep_phases=["run_terminal", "pre_llm"],
        )
        result = apply_state_retention(state, policy=policy)
        assert "checkpoint:run_1:5:run_terminal" in result
        assert "checkpoint:run_1:3:pre_llm" in result
        # some_custom_phase is not in always_keep_phases AND step 2 is not the latest step
        assert "checkpoint:run_1:2:some_custom_phase" not in result


class TestApplyStateRetentionKeepStatePrefixes:
    """keep_state_prefixes preserves matching keys."""

    def test_prefix_matching(self):
        state = {
            "checkpoint:run_1:latest": {"timestamp_ms": 100, "step": 1, "phase": "pre_llm"},
            "myapp:settings:color": "blue",
            "myapp:settings:font": "monospace",
            "other:key": 42,
        }
        policy = StateRetentionPolicy(
            max_runs=10,
            keep_state_prefixes=["myapp:settings:"],
        )
        result = apply_state_retention(state, policy=policy)
        assert "myapp:settings:color" in result
        assert "myapp:settings:font" in result
        # "other:key" is passthrough, so it is also kept
        assert "other:key" in result


class TestApplyStateRetentionEffectKeysWithinBudget:
    """Effect keys within effect budget are kept."""

    def test_effects_within_budget(self):
        state = {
            "checkpoint:run_1:latest": {"timestamp_ms": 100, "step": 5, "phase": "pre_llm"},
            "effect:run_1:5:eff_1": {"data": "a"},
            "effect:run_1:5:eff_2": {"data": "b"},
        }
        policy = StateRetentionPolicy(max_runs=10, max_effect_entries_per_run=10)
        result = apply_state_retention(state, policy=policy)
        assert "effect:run_1:5:eff_1" in result
        assert "effect:run_1:5:eff_2" in result


class TestApplyStateRetentionEffectKeysBeyondBudget:
    """Effect keys beyond budget are dropped."""

    def test_effects_beyond_budget(self):
        state = {
            "checkpoint:run_1:latest": {"timestamp_ms": 100, "step": 5, "phase": "pre_llm"},
        }
        # Add many effect keys
        for i in range(10):
            state[f"effect:run_1:{i}:eff_{i}"] = {"data": f"val_{i}"}

        policy = StateRetentionPolicy(max_runs=10, max_effect_entries_per_run=3)
        result = apply_state_retention(state, policy=policy)
        effect_keys = [k for k in result if k.startswith("effect:")]
        assert len(effect_keys) == 3

    def test_newest_effects_kept(self):
        state = {
            "checkpoint:run_1:latest": {"timestamp_ms": 100, "step": 5, "phase": "pre_llm"},
            "effect:run_1:1:eff_a": {"data": "old"},
            "effect:run_1:2:eff_b": {"data": "mid"},
            "effect:run_1:9:eff_c": {"data": "new"},
        }
        policy = StateRetentionPolicy(max_runs=10, max_effect_entries_per_run=2)
        result = apply_state_retention(state, policy=policy)
        effect_keys = [k for k in result if k.startswith("effect:")]
        # The sorting is by (-step, key), so step=9 and step=2 are newest
        assert len(effect_keys) == 2
        assert "effect:run_1:9:eff_c" in result
        assert "effect:run_1:2:eff_b" in result


# ===========================================================================
# 3. compact_thread_memory
# ===========================================================================


class TestCompactThreadMemoryBasic:
    """Basic compaction with events and no state changes."""

    def test_basic_compaction(self):
        async def _run():
            store = InMemoryMemoryStore()
            await store.setup()
            for i in range(10):
                await store.append_event(_event(i, "message"))

            result = await compact_thread_memory(
                store,
                thread_id="t1",
                event_policy=RetentionPolicy(
                    max_events_per_thread=5, keep_event_types=[]
                ),
                state_policy=StateRetentionPolicy(),
            )
            await store.close()
            return result

        result = run_async(_run())
        assert isinstance(result, MemoryCompactionResult)
        assert result.events_before == 10
        assert result.events_after == 5
        assert result.events_removed == 5
        assert result.state_keys_before == 0
        assert result.state_keys_after == 0
        assert result.state_keys_removed == 0
        assert result.state_keys_removed_effective == 0


class TestCompactThreadMemoryReplaceNotImplemented:
    """Compaction with NotImplementedError on replace_thread_events (skips gracefully)."""

    def test_replace_not_implemented(self):
        async def _run():
            store = InMemoryMemoryStore()
            await store.setup()
            for i in range(10):
                await store.append_event(_event(i, "message"))

            # Monkey-patch to raise NotImplementedError
            original_replace = store.replace_thread_events

            async def _raise_not_impl(thread_id, events):
                raise NotImplementedError("not supported")

            store.replace_thread_events = _raise_not_impl  # type: ignore[assignment]

            result = await compact_thread_memory(
                store,
                thread_id="t1",
                event_policy=RetentionPolicy(
                    max_events_per_thread=3, keep_event_types=[]
                ),
                state_policy=StateRetentionPolicy(),
            )

            # Events should still be in original form since replace failed
            remaining = await store.get_recent_events("t1", limit=100)
            await store.close()
            return result, len(remaining)

        result, remaining_count = run_async(_run())
        # The result still reports the compaction intent
        assert result.events_before == 10
        assert result.events_after == 3
        assert result.events_removed == 7
        # But the events were not actually replaced
        assert remaining_count == 10


class TestCompactThreadMemoryDeleteStateNotImplemented:
    """Compaction with NotImplementedError on delete_state (stops deletion with break)."""

    def test_delete_state_not_implemented_breaks(self):
        async def _run():
            store = InMemoryMemoryStore()
            await store.setup()

            # Add state: two runs, but max_runs=1 so the old one should be pruned
            await store.put_state(
                "t1",
                "checkpoint:run_new:latest",
                {"timestamp_ms": 200, "step": 2, "phase": "pre_llm"},
            )
            await store.put_state(
                "t1",
                "checkpoint:run_new:2:pre_llm",
                {"step": 2, "phase": "pre_llm"},
            )
            await store.put_state(
                "t1",
                "checkpoint:run_old:latest",
                {"timestamp_ms": 10, "step": 1, "phase": "pre_llm"},
            )
            await store.put_state(
                "t1",
                "checkpoint:run_old:1:pre_llm",
                {"step": 1, "phase": "pre_llm"},
            )

            # Monkey-patch delete_state to raise NotImplementedError
            async def _raise_not_impl(thread_id, key):
                raise NotImplementedError("not supported")

            store.delete_state = _raise_not_impl  # type: ignore[assignment]

            result = await compact_thread_memory(
                store,
                thread_id="t1",
                event_policy=RetentionPolicy(max_events_per_thread=10000),
                state_policy=StateRetentionPolicy(max_runs=1),
            )
            await store.close()
            return result

        result = run_async(_run())
        # state_keys_removed indicates what *should* be removed
        assert result.state_keys_removed > 0
        # But effective removal is 0 because the first delete raised and broke
        assert result.state_keys_removed_effective == 0


class TestCompactThreadMemoryResultCounts:
    """Returns correct MemoryCompactionResult counts."""

    def test_full_counts(self):
        async def _run():
            store = InMemoryMemoryStore()
            await store.setup()

            # 6 events
            for i in range(6):
                await store.append_event(_event(i, "message"))

            # State with 2 runs; we keep only 1
            await store.put_state(
                "t1",
                "checkpoint:run_new:latest",
                {"timestamp_ms": 200, "step": 3, "phase": "runtime_state"},
            )
            await store.put_state(
                "t1",
                "checkpoint:run_new:3:runtime_state",
                {"step": 3, "phase": "runtime_state"},
            )
            await store.put_state(
                "t1",
                "checkpoint:run_old:latest",
                {"timestamp_ms": 10, "step": 1, "phase": "runtime_state"},
            )
            await store.put_state(
                "t1",
                "checkpoint:run_old:1:runtime_state",
                {"step": 1, "phase": "runtime_state"},
            )
            await store.put_state(
                "t1",
                "effect:run_new:3:tc_1",
                {"success": True},
            )
            await store.put_state(
                "t1",
                "effect:run_old:1:tc_1",
                {"success": True},
            )
            # passthrough key
            await store.put_state("t1", "custom:keep_me", {"value": 1})

            result = await compact_thread_memory(
                store,
                thread_id="t1",
                event_policy=RetentionPolicy(
                    max_events_per_thread=4, keep_event_types=[]
                ),
                state_policy=StateRetentionPolicy(max_runs=1),
            )
            remaining_events = await store.get_recent_events("t1", limit=100)
            remaining_state = await store.list_state("t1")
            await store.close()
            return result, remaining_events, remaining_state

        result, remaining_events, remaining_state = run_async(_run())

        assert result.events_before == 6
        assert result.events_after == 4
        assert result.events_removed == 2

        # Verify events were actually replaced
        assert len(remaining_events) == 4

        # State: 7 keys total. run_old:latest, run_old:1:runtime_state, effect:run_old:1:tc_1
        # should be removed (3 keys removed)
        assert result.state_keys_before == 7
        assert result.state_keys_removed > 0
        assert result.state_keys_removed_effective == result.state_keys_removed
        assert result.state_keys_after + result.state_keys_removed == result.state_keys_before

        # Verify remaining state
        assert "checkpoint:run_new:latest" in remaining_state
        assert "custom:keep_me" in remaining_state
        assert "checkpoint:run_old:latest" not in remaining_state


# ===========================================================================
# 4. Private helpers
# ===========================================================================


class TestSafeInt:
    """_safe_int() behavior."""

    def test_int_passthrough(self):
        assert _safe_int(42) == 42
        assert _safe_int(0) == 0
        assert _safe_int(-5) == -5

    def test_float_to_int(self):
        assert _safe_int(3.7) == 3
        assert _safe_int(0.0) == 0
        assert _safe_int(-2.9) == -2

    def test_digit_string_to_int(self):
        assert _safe_int("123") == 123
        assert _safe_int(" 456 ") == 456
        assert _safe_int("0") == 0

    def test_negative_string_returns_int(self):
        assert _safe_int("-1") == -1
        assert _safe_int("-99") == -99

    def test_none_returns_none(self):
        assert _safe_int(None) is None

    def test_non_digit_string_returns_none(self):
        assert _safe_int("abc") is None
        assert _safe_int("12.5") is None
        assert _safe_int("") is None
        assert _safe_int(" ") is None


class TestExtractTimestampMs:
    """_extract_timestamp_ms() behavior."""

    def test_dict_with_timestamp_ms(self):
        assert _extract_timestamp_ms({"timestamp_ms": 12345}) == 12345

    def test_dict_without_timestamp_ms(self):
        assert _extract_timestamp_ms({"other_key": 99}) == 0

    def test_non_dict_returns_zero(self):
        assert _extract_timestamp_ms("string") == 0
        assert _extract_timestamp_ms(42) == 0
        assert _extract_timestamp_ms(None) == 0
        assert _extract_timestamp_ms([1, 2, 3]) == 0

    def test_dict_with_string_timestamp(self):
        assert _extract_timestamp_ms({"timestamp_ms": "999"}) == 999

    def test_dict_with_none_timestamp(self):
        assert _extract_timestamp_ms({"timestamp_ms": None}) == 0


class TestParseCheckpointLatestKey:
    """_parse_checkpoint_latest_key() behavior."""

    def test_valid_key(self):
        assert _parse_checkpoint_latest_key("checkpoint:run_123:latest") == "run_123"

    def test_valid_key_with_special_chars(self):
        assert _parse_checkpoint_latest_key("checkpoint:abc-def:latest") == "abc-def"

    def test_invalid_not_checkpoint_prefix(self):
        assert _parse_checkpoint_latest_key("other:run_1:latest") is None

    def test_invalid_not_latest_suffix(self):
        assert _parse_checkpoint_latest_key("checkpoint:run_1:oldest") is None

    def test_invalid_too_few_parts(self):
        assert _parse_checkpoint_latest_key("checkpoint:latest") is None

    def test_invalid_too_many_parts(self):
        assert _parse_checkpoint_latest_key("checkpoint:run_1:latest:extra") is None

    def test_invalid_empty_run_id(self):
        assert _parse_checkpoint_latest_key("checkpoint::latest") is None

    def test_whitespace_run_id(self):
        assert _parse_checkpoint_latest_key("checkpoint: :latest") is None


class TestParseCheckpointStateKey:
    """_parse_checkpoint_state_key() behavior."""

    def test_valid_key(self):
        result = _parse_checkpoint_state_key("checkpoint:run_1:5:pre_llm")
        assert result == ("run_1", 5, "pre_llm")

    def test_valid_key_with_colon_in_phase(self):
        # split(":", 3) means phase can contain colons
        result = _parse_checkpoint_state_key("checkpoint:run_1:5:phase:extra")
        assert result == ("run_1", 5, "phase:extra")

    def test_missing_parts(self):
        assert _parse_checkpoint_state_key("checkpoint:run_1") is None
        assert _parse_checkpoint_state_key("checkpoint:run_1:5") is None

    def test_not_checkpoint_prefix(self):
        assert _parse_checkpoint_state_key("other:run_1:5:pre_llm") is None

    def test_empty_run_id(self):
        assert _parse_checkpoint_state_key("checkpoint::5:pre_llm") is None

    def test_non_int_step(self):
        assert _parse_checkpoint_state_key("checkpoint:run_1:abc:pre_llm") is None

    def test_empty_phase(self):
        assert _parse_checkpoint_state_key("checkpoint:run_1:5:") is None

    def test_whitespace_phase(self):
        assert _parse_checkpoint_state_key("checkpoint:run_1:5: ") is None


class TestParseEffectKey:
    """_parse_effect_key() behavior."""

    def test_valid_key(self):
        result = _parse_effect_key("effect:run_1:5:tc_abc")
        assert result == ("run_1", 5)

    def test_valid_key_with_colon_in_id(self):
        result = _parse_effect_key("effect:run_1:5:tc:with:colons")
        assert result == ("run_1", 5)

    def test_invalid_not_effect_prefix(self):
        assert _parse_effect_key("checkpoint:run_1:5:tc_abc") is None

    def test_invalid_too_few_parts(self):
        assert _parse_effect_key("effect:run_1:5") is None
        assert _parse_effect_key("effect:run_1") is None

    def test_invalid_empty_run_id(self):
        assert _parse_effect_key("effect::5:tc_abc") is None

    def test_invalid_non_int_step(self):
        assert _parse_effect_key("effect:run_1:abc:tc_abc") is None
