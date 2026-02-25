from afk.core import Runner
from afk.agents import Agent
from afk.tools import ToolRegistry, tool

from afk.agents import Agent
from afk.tools import tool
from afk.core import Runner

from pydantic import BaseModel
import asyncio 
class WeatherArgs(BaseModel):
    city: str

@tool(args_model=WeatherArgs, name="get_weather", description="Get current weather for a city.")
def get_weather(args: WeatherArgs) -> dict:
    return {"city": args.city, "temp_f": 72, "condition": "sunny"}

agent = Agent(
    name="weather-bot",
    model="ollama_chat/gpt-oss:20b",
    instructions="Answer weather questions using your tools.",
    tools=[get_weather],           # ← Attach tools here
    reasoning_effort="high",
    reasoning_enabled=True,
)

async def main():
    runner = Runner()
    result = await runner.run_stream(agent, user_message="What's the weather in Austin?")

    async for r in result: 
        if r.type== "tool_started":
            print(f"Agent is calling tool: {r.tool_name} with args: {r.data}")
        elif r.type == "tool_completed":
            print(f"Agent got tool response: {r.tool_output}")
        elif r.type == "text_delta":
            print(r.text_delta, end="", flush=True)
        else:
            print(f"Received unknown result type: {r.type}")

asyncio.run(main())