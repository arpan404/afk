"""LLMBuilder and middleware configuration for enterprise prompting."""

from dataclasses import replace

from afk.llms import LLMBuilder, LLMRequest, LLMResponse, Message, MiddlewareStack

from .models import OpportunityForecast


async def attach_tenant_context(call_next, req: LLMRequest) -> LLMResponse:
    """Inject tenant metadata into every request."""
    metadata = dict(req.metadata)
    metadata["tenant"] = "north-america-enterprise"
    metadata["use_case"] = "pipeline_forecast"
    return await call_next(replace(req, metadata=metadata))


async def enforce_json_response(call_next, req: LLMRequest) -> LLMResponse:
    """Add a hard JSON system instruction for stable structured parsing."""
    instruction = Message(
        role="system",
        content=(
            "Return strictly valid JSON that matches the requested schema. "
            "No markdown, no prose."
        ),
    )
    return await call_next(replace(req, messages=[instruction, *req.messages]))


def build_client():
    """Construct a profile-based LLM client with middleware instrumentation."""
    return (
        LLMBuilder()
        .provider("litellm")
        .model("gpt-4.1-mini")
        .for_agent_runtime()
        .with_middlewares(
            MiddlewareStack(
                chat=[attach_tenant_context, enforce_json_response],
            )
        )
        .build()
    )


async def forecast_opportunity(prompt: str) -> OpportunityForecast:
    """Generate structured forecast output for one opportunity narrative."""
    client = build_client()
    req = LLMRequest(
        model="gpt-4.1-mini",
        messages=[Message(role="user", content=prompt)],
        thinking=True,
        thinking_effort="balanced",
        max_thinking_tokens=256,
    )
    response = await client.chat(req, response_model=OpportunityForecast)
    payload = response.structured_response or {}
    return OpportunityForecast.model_validate(payload)
