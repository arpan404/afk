"""
---
name: Production Agent — Agents
description: Agent definitions with the coordinator, specialist subagents, and dynamic InstructionProvider.
tags: [agent, subagents, delegation, instructions]
---
---
This module defines the agent hierarchy for the production task manager. The coordinator agent handles task operations directly and delegates analytical and planning queries to specialist subagents. A dynamic InstructionProvider personalizes the coordinator's instructions based on runtime context (like time of day). This pattern shows how to build modular, specialized agent systems.
---
"""

from afk.agents import Agent  # <- Agent is the main building block. We create multiple agents that work together.

from config import fail_safe_config  # <- Import the shared FailSafeConfig from config.py.
from tools import registry  # <- Import the ToolRegistry with all tools registered, plus policy and middleware.


# ===========================================================================
# Dynamic InstructionProvider
# ===========================================================================

def task_manager_instructions(context: dict) -> str:  # <- InstructionProvider is a callable that takes a context dict and returns an instruction string. The runner calls this at the start of each run, so instructions can adapt to runtime conditions.
    """Dynamic instruction provider that adapts based on runtime context.

    The runner passes the run context to this function, which can include
    user preferences, time of day, feature flags, or any other runtime data.
    This makes the agent's behavior configurable without changing code.
    """
    base_instructions = """
    You are a production-ready task management assistant. You help users manage
    their tasks efficiently with full CRUD operations.

    Your capabilities:
    - Create tasks with add_task (title, priority, category)
    - List tasks with list_tasks (filter by status and priority)
    - Mark tasks done with complete_task
    - Update task properties with update_task
    - Delete tasks with delete_task
    - Show statistics with get_stats

    When delegating:
    - For analytical questions (stats, trends, productivity), delegate to the analyst subagent.
    - For planning questions (prioritization, scheduling, strategy), delegate to the planner subagent.

    Interaction rules:
    - Be concise and action-oriented
    - Always confirm what action was taken
    - Show task IDs so the user can reference them
    - Proactively suggest next steps after completing an action
    """

    # Adapt instructions based on context
    user_name = context.get("user_name", "")  # <- context is a dict that can contain any JSON-serializable data. The runner passes it from runner.run(..., context={...}).
    if user_name:
        base_instructions += f"\n    Address the user as {user_name}."

    mode = context.get("mode", "")
    if mode == "verbose":
        base_instructions += "\n    Provide detailed explanations for every action."
    elif mode == "brief":
        base_instructions += "\n    Keep responses as short as possible — one line per action."

    return base_instructions


# ===========================================================================
# Specialist subagents
# ===========================================================================
# These subagents handle specific types of queries. The coordinator delegates
# to them when the user asks for analysis or planning rather than direct
# task operations.

analyst_agent = Agent(
    name="task-analyst",  # <- A subagent focused on analyzing task data and productivity patterns.
    model="ollama_chat/gpt-oss:20b",  # <- Same model for subagents. You can use different models for different specialists (e.g., a smaller model for simple tasks).
    instructions="""You are a task analytics specialist. When given task data, you provide:
    - Clear summaries of task distribution (by priority, category, status)
    - Productivity insights (completion rates, trends)
    - Actionable recommendations for improving task management
    - Highlights of overdue or high-priority items that need attention

    Be data-driven and concise. Use numbers and percentages when possible.""",  # <- Focused instructions make subagents more effective than one large prompt.
    tools=registry.list(),  # <- The analyst gets the same tools so it can look up task data. In a real system, you might give subagents a subset of tools.
)

planner_agent = Agent(
    name="task-planner",  # <- A subagent focused on task prioritization and scheduling strategy.
    model="ollama_chat/gpt-oss:20b",
    instructions="""You are a task planning specialist. When given tasks and context, you provide:
    - Smart prioritization recommendations based on urgency and importance
    - Suggested order for tackling open tasks
    - Time management advice
    - Risk assessment for overloaded categories or neglected priorities

    Think like a project manager. Be practical and actionable.""",
    tools=registry.list(),
)


# ===========================================================================
# Coordinator agent (the one users interact with)
# ===========================================================================

coordinator = Agent(
    name="task-manager",  # <- The top-level agent the user interacts with directly.
    model="ollama_chat/gpt-oss:20b",  # <- The llm model the agent will use.
    instructions=task_manager_instructions,  # <- Dynamic InstructionProvider! Instead of a static string, we pass a callable that generates instructions based on runtime context. The runner calls this function at the start of each run.
    tools=registry.list(),  # <- Pull tools from the ToolRegistry. The registry manages all tools centrally -- policy and middleware are applied automatically.
    subagents=[analyst_agent, planner_agent],  # <- Enable delegation! The coordinator can route queries to the analyst or planner subagents. The Runner handles the routing automatically based on the coordinator's instructions.
    fail_safe=fail_safe_config,  # <- Attach the FailSafeConfig for execution safety limits.
)
