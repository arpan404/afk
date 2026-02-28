"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Module: runtime/hedging.py.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def run_with_hedge(
    primary: Callable[[], Awaitable[T]],
    secondary: Callable[[], Awaitable[T]] | None,
    *,
    delay_s: float,
) -> T:
    """Run primary request and optionally hedge with delayed secondary."""
    primary_task = asyncio.create_task(primary())
    if secondary is None:
        return await primary_task

    async def _delayed_secondary() -> T:
        await asyncio.sleep(max(0.0, delay_s))
        return await secondary()

    secondary_task = asyncio.create_task(_delayed_secondary())

    done, pending = await asyncio.wait(
        {primary_task, secondary_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    winner = done.pop()

    # If the first-completed task raised an exception and the other is still
    # running, give the other task a chance to succeed instead of immediately
    # propagating the error.
    if winner.exception() is not None and pending:
        try:
            done2, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            fallback = done2.pop()
            if fallback.exception() is None:
                winner = fallback
        except Exception:
            pass
        # Cancel anything still pending after the fallback attempt.
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    else:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    return await winner
