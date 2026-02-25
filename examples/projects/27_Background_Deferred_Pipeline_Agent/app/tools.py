"""Background-capable tools for deferred pipeline execution."""

import asyncio

from pydantic import BaseModel

from afk.tools import ToolDeferredHandle, ToolResult, tool


class NoArgs(BaseModel):
    pass


@tool(args_model=NoArgs, name="compile_report")
async def compile_report(args: NoArgs) -> ToolResult[dict[str, str]]:
    _ = args
    loop = asyncio.get_running_loop()
    report_future: asyncio.Future[dict[str, str]] = loop.create_future()

    async def finish_work() -> None:
        await asyncio.sleep(0.02)
        if not report_future.done():
            report_future.set_result(
                {
                    "status": "ok",
                    "artifact": "reports/weekly_health.md",
                }
            )

    asyncio.create_task(finish_work())
    return ToolResult(
        output=None,
        success=True,
        deferred=ToolDeferredHandle(
            ticket_id="report-ticket-27",
            tool_name="compile_report",
            status="running",
            summary="Background report build started",
            resume_hint="Continue drafting summary while report compiles",
            poll_after_s=0.01,
        ),
        metadata={"background_future": report_future},
    )


@tool(args_model=NoArgs, name="draft_executive_summary")
def draft_executive_summary(args: NoArgs) -> dict[str, str]:
    _ = args
    return {"status": "drafted"}
