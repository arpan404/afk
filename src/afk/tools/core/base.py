"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

This module defines the types and base classes for tools that can be registered and used by the AFK agent.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import (
    Any,
    Generic,
    Literal,
    TypeVar,
    Union,
)

from pydantic import BaseModel, ValidationError

from afk.tools.core.errors import (
    ToolExecutionError,
    ToolTimeoutError,
    ToolValidationError,
)

ArgsT = TypeVar("ArgsT", bound=BaseModel)
ReturnT = TypeVar("ReturnT")

AsyncToolFn = Callable[..., Awaitable[Any]]
SyncToolFn = Callable[..., Any]
ToolFn = Union[AsyncToolFn, SyncToolFn]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """
    Stable tool metadata used for registry listing + model-facing export.
    """

    name: str
    description: str
    parameters_schema: dict[str, Any]  # JSON Schema for the tool's arguments


@dataclass(frozen=True, slots=True)
class ToolContext:
    """
    Contextual information available to a tool during its execution.
    Keep it simple and serializable, as it may be passed to tools running in different environments or processes.
    """

    request_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolDeferredHandle:
    """
    Deferred tool execution marker used for background orchestration.
    """

    ticket_id: str
    tool_name: str
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    poll_after_s: float | None = None
    summary: str | None = None
    resume_hint: str | None = None


@dataclass(frozen=True, slots=True)
class ToolResult(Generic[ReturnT]):
    """
    Standardized result object returned by tools after execution.
    Contains the output of the tool, as well as any relevant metadata about the execution.

    Note: this is used by the hooks and middlewares to have a standard way of representing the result of a tool execution,
    which allows for easier chaining and error handling.
    """

    output: ReturnT | None = None
    success: bool = True
    error_message: str | None = None
    tool_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_call_id: str | None = None
    deferred: ToolDeferredHandle | None = None


def as_async(fn: ToolFn) -> AsyncToolFn:
    """
    Utility function to convert a synchronous function into an asynchronous one.
    This allows the tool registry and execution logic to treat all tools as async, simplifying the implementation.
    """
    if asyncio.iscoroutinefunction(fn):
        return fn  # type: ignore[return-value]

    async def _wrapped(*args: Any, **kwargs: Any) -> Any:
        # run sync function in threadpool
        return await asyncio.to_thread(fn, *args, **kwargs)

    return _wrapped


def _infer_call_style(fn: Callable[..., Any]) -> str:
    """
    Determine how to call a tool/hook based on the signature.

    Allowed for tools/hooks:
      (args)
      (args, ctx)
      (ctx, args)

    We accept ctx by name "ctx" OR annotation ToolContext.
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    if any(p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL) for p in params):
        raise ToolValidationError(
            f"Tool function '{getattr(fn, '__name__', 'unknown')}' cannot have *args or **kwargs."
        )

    if len(params) == 1:
        return "args"

    if len(params) == 2:
        p0, p1 = params

        if p0.annotation is ToolContext or p0.name == "ctx":
            return "ctx_args"

        if p1.annotation is ToolContext or p1.name == "ctx":
            return "args_ctx"

        # If user didn't annotate ctx, default to (args, ctx) only if second param is named ctx.
        # Otherwise treat as invalid to avoid accidental miscalls.
        raise ToolValidationError(
            f"Tool function '{getattr(fn, '__name__', 'unknown')}' must include ToolContext "
            f"as 'ctx' (by name or annotation). Signature: {sig}"
        )

    raise ToolValidationError(
        f"Tool function '{getattr(fn, '__name__', 'unknown')}' has invalid signature. "
        f"Expected (args) or (args, ctx) or (ctx, args). Got {sig}."
    )


def _infer_middleware_style(fn: Callable[..., Any]) -> str:
    """
    Middleware signature inference.

    Allowed:
      (call_next, args)
      (call_next, args, ctx)
      (args, ctx, call_next)
      (ctx, args, call_next)

    call_next is a callable: async (args, ctx) -> output
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    if any(p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL) for p in params):
        raise ToolValidationError(
            f"Middleware '{getattr(fn, '__name__', 'unknown')}' cannot have *args or **kwargs."
        )

    if len(params) == 2:
        return "next_args"

    if len(params) == 3:
        p0, p1, p2 = params
        # common patterns:
        if p0.name in ("call_next", "next") or "next" in p0.name:
            return "next_args_ctx"
        if p2.name in ("call_next", "next") or "next" in p2.name:
            # could be (args, ctx, next) or (ctx, args, next)
            if p0.annotation is ToolContext or p0.name == "ctx":
                return "ctx_args_next"
            return "args_ctx_next"

        # fallback: assume (args, ctx, call_next) if middle param looks like ctx
        if p1.annotation is ToolContext or p1.name == "ctx":
            return "args_ctx_next"

        raise ToolValidationError(
            f"Middleware '{getattr(fn, '__name__', 'unknown')}' must follow a supported signature. Got {sig}"
        )

    raise ToolValidationError(
        f"Middleware '{getattr(fn, '__name__', 'unknown')}' has invalid signature length. Got {sig}"
    )


class BaseTool(Generic[ArgsT, ReturnT]):
    """
    Base class for defining a function-based tool/hook.

    IMPORTANT: BaseTool.call returns ToolResult and does NOT throw tool errors by default,
    because hooks/middleware chains rely on ToolResult.success.
    """

    def __init__(
        self,
        *,
        spec: ToolSpec,
        fn: ToolFn,
        args_model: type[ArgsT],
        default_timeout: float | None = None,
        raise_on_error: bool = False,
    ) -> None:
        self.spec = spec
        self._original_fn = fn
        self.fn = as_async(fn)
        self.args_model = args_model
        self.default_timeout = default_timeout
        self.raise_on_error = raise_on_error

        self._call_style = _infer_call_style(fn)

    def validate(self, raw_args: dict[str, Any]) -> ArgsT:
        try:
            return self.args_model.model_validate(raw_args)
        except ValidationError as e:
            raise ToolValidationError(
                f"Invalid arguments for tool '{self.spec.name}': {e}"
            ) from e

    async def _invoke(self, args: ArgsT, ctx: ToolContext) -> Any:
        if self._call_style == "args":
            return await self.fn(args)
        if self._call_style == "args_ctx":
            return await self.fn(args, ctx)
        return await self.fn(ctx, args)

    async def call(
        self,
        raw_args: dict[str, Any],
        *,
        ctx: ToolContext | None = None,
        timeout: float | None = None,
        tool_call_id: str | None = None,
    ) -> ToolResult[ReturnT]:
        ctx = ctx or ToolContext()

        try:
            args = self.validate(raw_args)
        except Exception as e:
            if self.raise_on_error:
                raise
            return ToolResult(
                output=None,
                success=False,
                error_message=str(e),
                tool_name=self.spec.name,
                tool_call_id=tool_call_id,
            )

        async def _run() -> Any:
            return await self._invoke(args, ctx)

        effective_timeout = timeout if timeout is not None else self.default_timeout

        try:
            if effective_timeout is not None:
                output = await asyncio.wait_for(_run(), timeout=effective_timeout)
            else:
                output = await _run()

            if isinstance(output, ToolResult):
                return ToolResult(
                    output=output.output,
                    success=output.success,
                    error_message=output.error_message,
                    tool_name=output.tool_name or self.spec.name,
                    metadata=dict(output.metadata),
                    tool_call_id=output.tool_call_id or tool_call_id,
                    deferred=output.deferred,
                )

            return ToolResult(
                output=output,
                success=True,
                error_message=None,
                tool_name=self.spec.name,
                tool_call_id=tool_call_id,
            )

        except TimeoutError:
            err = ToolTimeoutError(
                f"Tool '{self.spec.name}' execution exceeded timeout of {effective_timeout} seconds."
            )
            if self.raise_on_error:
                raise err
            return ToolResult(
                output=None,
                success=False,
                error_message=str(err),
                tool_name=self.spec.name,
                tool_call_id=tool_call_id,
            )

        except Exception as e:
            err = ToolExecutionError(f"Error executing tool '{self.spec.name}': {e}")
            if self.raise_on_error:
                raise err
            return ToolResult(
                output=None,
                success=False,
                error_message=str(err),
                tool_name=self.spec.name,
                tool_call_id=tool_call_id,
            )


class PreHook(BaseTool[ArgsT, Any]):
    """
    Prehooks are executed before the main tool execution.
    They can transform args. Output must be a dict compatible with the main tool's args model
    (unless you design a richer typed pipeline later).
    """


class PostHook(BaseTool[BaseModel, Any]):
    """
    Post-hooks are executed after the main tool execution.

    Recommended: use an args_model with at least {"output": Any} so posthooks can read/modify it.
    """


class Middleware(Generic[ArgsT, ReturnT]):
    """
    Middleware wraps the tool execution.

    Middleware fn can be sync or async. Supported signatures:
      (call_next, args)
      (call_next, args, ctx)
      (args, ctx, call_next)
      (ctx, args, call_next)

    call_next: async (args, ctx) -> output
    """

    def __init__(
        self, *, spec: ToolSpec, fn: ToolFn, default_timeout: float | None = None
    ) -> None:
        self.spec = spec
        self._original_fn = fn
        self.fn = as_async(fn)
        self.default_timeout = default_timeout
        self._mw_style = _infer_middleware_style(fn)

    async def call(
        self,
        call_next: Callable[[ArgsT, ToolContext], Awaitable[ReturnT]],
        args: ArgsT,
        ctx: ToolContext,
        *,
        timeout: float | None = None,
    ) -> ReturnT:
        async def _run() -> ReturnT:
            if self._mw_style == "next_args":
                return await self.fn(call_next, args)
            if self._mw_style == "next_args_ctx":
                return await self.fn(call_next, args, ctx)
            if self._mw_style == "ctx_args_next":
                return await self.fn(ctx, args, call_next)
            # "args_ctx_next"
            return await self.fn(args, ctx, call_next)

        effective_timeout = timeout if timeout is not None else self.default_timeout
        try:
            if effective_timeout is not None:
                return await asyncio.wait_for(_run(), timeout=effective_timeout)
            return await _run()
        except TimeoutError as e:
            raise ToolTimeoutError(
                f"Middleware '{self.spec.name}' exceeded timeout of {effective_timeout} seconds."
            ) from e


class Tool(BaseTool[ArgsT, ReturnT]):
    """
    Main Tool class supporting prehooks, posthooks, and middleware wrapping.
    """

    def __init__(
        self,
        *,
        spec: ToolSpec,
        fn: ToolFn,
        args_model: type[ArgsT],
        default_timeout: float | None = None,
        prehooks: list[PreHook[Any, Any]] | None = None,
        posthooks: list[PostHook] | None = None,
        middlewares: list[Middleware[ArgsT, ReturnT]] | None = None,
        raise_on_error: bool = False,
    ) -> None:
        super().__init__(
            spec=spec,
            fn=fn,
            args_model=args_model,
            default_timeout=default_timeout,
            raise_on_error=raise_on_error,
        )
        self.prehooks = prehooks or []
        self.posthooks = posthooks or []
        self.middlewares = middlewares or []

    async def call(
        self,
        raw_args: dict[str, Any],
        *,
        ctx: ToolContext | None = None,
        timeout: float | None = None,
        tool_call_id: str | None = None,
    ) -> ToolResult[ReturnT]:
        ctx = ctx or ToolContext()

        # 1) Validate initial args
        try:
            args = self.validate(raw_args)
        except Exception as e:
            if self.raise_on_error:
                raise
            return ToolResult(
                output=None,
                success=False,
                error_message=str(e),
                tool_name=self.spec.name,
                tool_call_id=tool_call_id,
            )

        # 2) Run prehooks (each must output dict compatible with main args_model)
        for pre in self.prehooks:
            pre_res = await pre.call(
                raw_args=args.model_dump(),
                ctx=ctx,
                timeout=pre.default_timeout,
                tool_call_id=tool_call_id,
            )
            if not pre_res.success:
                if self.raise_on_error:
                    raise ToolExecutionError(
                        f"Pre-hook '{pre.spec.name}' failed: {pre_res.error_message}"
                    )
                return ToolResult(
                    output=None,
                    success=False,
                    error_message=f"Pre-hook '{pre.spec.name}' failed: {pre_res.error_message}",
                    tool_name=self.spec.name,
                    tool_call_id=tool_call_id,
                )

            if not isinstance(pre_res.output, dict):
                msg = f"Pre-hook '{pre.spec.name}' must return a dict of args; got {type(pre_res.output).__name__}"
                if self.raise_on_error:
                    raise ToolExecutionError(msg)
                return ToolResult(
                    output=None,
                    success=False,
                    error_message=msg,
                    tool_name=self.spec.name,
                    tool_call_id=tool_call_id,
                )

            # Re-validate transformed args against the MAIN tool args_model
            try:
                args = self.args_model.model_validate(pre_res.output)
            except ValidationError as e:
                msg = f"Pre-hook '{pre.spec.name}' produced invalid args for '{self.spec.name}': {e}"
                if self.raise_on_error:
                    raise ToolValidationError(msg)
                return ToolResult(
                    output=None,
                    success=False,
                    error_message=msg,
                    tool_name=self.spec.name,
                    tool_call_id=tool_call_id,
                )

        # 3) Build middleware chain around the core invoke
        async def core_call(a: ArgsT, c: ToolContext) -> ReturnT:
            # respects tool signature (args) / (args, ctx) / (ctx, args)
            return await self._invoke(a, c)  # type: ignore[return-value]

        wrapped = core_call
        # Apply middlewares in reverse registration order (last added wraps closest to core)
        for mw in reversed(self.middlewares):
            prev = wrapped

            async def _wrapped(a: ArgsT, c: ToolContext, _mw=mw, _prev=prev) -> ReturnT:
                return await _mw.call(_prev, a, c, timeout=_mw.default_timeout)

            wrapped = _wrapped

        # 4) Run tool (with timeout)
        async def run_tool() -> ReturnT:
            return await wrapped(args, ctx)

        effective_timeout = timeout if timeout is not None else self.default_timeout

        try:
            if effective_timeout is not None:
                output = await asyncio.wait_for(run_tool(), timeout=effective_timeout)
            else:
                output = await run_tool()

        except TimeoutError:
            err = ToolTimeoutError(
                f"Tool '{self.spec.name}' execution exceeded timeout of {effective_timeout} seconds."
            )
            if self.raise_on_error:
                raise err
            return ToolResult(
                output=None,
                success=False,
                error_message=str(err),
                tool_name=self.spec.name,
                tool_call_id=tool_call_id,
            )

        except Exception as e:
            err = ToolExecutionError(f"Error executing tool '{self.spec.name}': {e}")
            if self.raise_on_error:
                raise err
            return ToolResult(
                output=None,
                success=False,
                error_message=str(err),
                tool_name=self.spec.name,
                tool_call_id=tool_call_id,
            )

        if isinstance(output, ToolResult):
            return ToolResult(
                output=output.output,
                success=output.success,
                error_message=output.error_message,
                tool_name=output.tool_name or self.spec.name,
                metadata=dict(output.metadata),
                tool_call_id=output.tool_call_id or tool_call_id,
                deferred=output.deferred,
            )

        # 5) Run posthooks
        # Posthooks get a dict payload by convention: {"output": ..., "tool_name": ...}
        post_payload: dict[str, Any] = {"output": output, "tool_name": self.spec.name}
        post_result: Any = output

        for post in self.posthooks:
            post_res = await post.call(
                raw_args=post_payload,
                ctx=ctx,
                timeout=post.default_timeout,
                tool_call_id=tool_call_id,
            )
            if not post_res.success:
                if self.raise_on_error:
                    raise ToolExecutionError(
                        f"Post-hook '{post.spec.name}' failed: {post_res.error_message}"
                    )
                return ToolResult(
                    output=None,
                    success=False,
                    error_message=f"Post-hook '{post.spec.name}' failed: {post_res.error_message}",
                    tool_name=self.spec.name,
                    tool_call_id=tool_call_id,
                )

            # posthook output becomes new "output"
            post_result = post_res.output
            post_payload = {"output": post_result, "tool_name": self.spec.name}

        return ToolResult(
            output=post_result,
            success=True,
            error_message=None,
            tool_name=self.spec.name,
            tool_call_id=tool_call_id,
        )
