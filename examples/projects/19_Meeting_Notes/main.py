"""
---
name: Meeting Notes
description: A meeting notes agent demonstrating Jinja2 prompt templates via instruction_file and prompts_dir, alongside callable InstructionProvider.
tags: [agent, runner, tools, instruction-file, prompts-dir, jinja2, context]
---
---
This example demonstrates two ways to create dynamic, context-aware agent instructions in AFK:

1. **instruction_file + prompts_dir** (Jinja2 templates): Load instructions from a Markdown file
   in the prompts/ directory. The file is a Jinja2 template that receives the runtime context
   as template variables. Use {{ meeting_type }}, {% if %}/{% elif %}/{% endif %}, {{ attendees | join(', ') }},
   and other Jinja2 syntax. The runner renders the template automatically before each run.

2. **Callable InstructionProvider**: Pass a Python function to instructions= that receives the
   runtime context dict and returns a string. More flexible than templates (you can do database
   lookups, complex logic, etc.) but templates are often simpler for text-generation patterns.

Both approaches receive the same runtime context passed via runner.run(context={...}). The template
approach is declarative and designer-friendly; the callable approach is imperative and developer-friendly.
This project uses the template-based agent (instruction_file) by default, with the callable agent
available for comparison.
---
"""

from pathlib import Path  # <- Path for resolving the prompts directory relative to this file.
from afk.core import Runner  # <- Runner executes agents and injects context into templates.
from afk.agents import Agent  # <- Agent defines the agent's model, instructions, and tools.

from tools import add_note, list_notes, summarize_notes, add_action_item, list_action_items  # <- Import tools from the tools module.


# ===========================================================================
# Approach 1: instruction_file + prompts_dir (Jinja2 templates)
# ===========================================================================
# This is the recommended approach for text-heavy, designer-friendly prompts.
# The template file lives in prompts/meeting_notes.md and uses Jinja2 syntax
# to adapt based on the runtime context (meeting_type, formality, attendees).
# The runner renders the template before each run, injecting context variables.

PROMPTS_DIR = Path(__file__).parent / "prompts"  # <- Resolve prompts directory relative to this file's location. This works regardless of where the script is launched from.

template_agent = Agent(
    name="meeting-notes-template",  # <- Agent using the template-based approach.
    model="ollama_chat/gpt-oss:20b",  # <- The LLM model the agent will use.
    instruction_file="meeting_notes.md",  # <- Path to the Jinja2 template file, resolved relative to prompts_dir. The runner loads this file, renders it with the runtime context, and uses the result as the agent's instructions.
    prompts_dir=PROMPTS_DIR,  # <- Directory containing prompt templates. Can also be set via the AFK_AGENT_PROMPTS_DIR environment variable. If omitted, defaults to .agents/prompt.
    context_defaults={  # <- Default context values injected into the template. These are used if no overrides are provided at runtime. Template variables {{ meeting_type }}, {{ formality }}, {{ attendees }} read from these.
        "meeting_type": "general",
        "formality": "casual",
        "attendees": [],
    },
    tools=[add_note, list_notes, summarize_notes, add_action_item, list_action_items],
)


# ===========================================================================
# Approach 2: Callable InstructionProvider (Python function)
# ===========================================================================
# This approach is more flexible — the function can do database lookups,
# complex branching, I/O, or anything else Python supports. Use this when
# your instruction logic is too complex for Jinja2 templates, or when you
# need async I/O to build the instructions.

def meeting_instructions(context: dict) -> str:  # <- InstructionProvider callable: receives context dict, returns instruction string. Called by the runner before each run.
    meeting_type = context.get("meeting_type", "general")
    formality = context.get("formality", "casual")
    attendees = context.get("attendees", [])

    type_instructions = {
        "standup": (
            "This is a daily standup meeting. Focus on three things per person:\n"
            "1. What did they do yesterday?\n"
            "2. What are they doing today?\n"
            "3. Any blockers?\n"
            "Keep notes brief and structured. Flag any blockers as action items immediately."
        ),
        "brainstorm": (
            "This is a brainstorming session. Capture ALL ideas without judgment.\n"
            "- Record every idea as a separate note.\n"
            "- Group related ideas when asked.\n"
            "- Do NOT evaluate — just capture."
        ),
        "review": (
            "This is a review meeting. Focus on:\n"
            "- What was presented and by whom.\n"
            "- Feedback given (positive and constructive).\n"
            "- Decisions made.\n"
            "- Follow-up items and owners."
        ),
        "planning": (
            "This is a planning meeting. Focus on:\n"
            "- Goals and objectives.\n"
            "- Tasks and estimated effort.\n"
            "- Dependencies and assignments.\n"
            "Create action items for every assigned task."
        ),
        "general": (
            "This is a general meeting. Take comprehensive notes covering:\n"
            "- Key discussion points.\n"
            "- Decisions made.\n"
            "- Action items and owners."
        ),
    }

    base = type_instructions.get(meeting_type, type_instructions["general"])
    tone = (
        "Use a professional, formal tone."
        if formality == "formal"
        else "Use a friendly, conversational tone."
    )
    attendee_note = ""
    if attendees:
        attendee_note = f"\nAttendees: {', '.join(attendees)}. Reference them by name."

    return (
        f"You are a meeting notes assistant.\n\n"
        f"Meeting type: {meeting_type}\n\n"
        f"{base}\n\n{tone}\n{attendee_note}\n\n"
        f"Use the available tools to record notes and action items. "
        f"When the user says 'summarize' or 'wrap up', provide a complete meeting summary."
    )


callable_agent = Agent(
    name="meeting-notes-callable",  # <- Agent using the callable approach.
    model="ollama_chat/gpt-oss:20b",
    instructions=meeting_instructions,  # <- Pass the callable directly. The runner invokes it with the runtime context on every run.
    context_defaults={
        "meeting_type": "general",
        "formality": "casual",
        "attendees": [],
    },
    tools=[add_note, list_notes, summarize_notes, add_action_item, list_action_items],
)


runner = Runner()


# ===========================================================================
# Interactive session
# ===========================================================================

if __name__ == "__main__":
    print("Meeting Notes Agent (type 'quit' to exit)")
    print("=" * 55)
    print()
    print("This example demonstrates two instruction approaches:")
    print("  1. instruction_file + prompts_dir (Jinja2 template)")
    print("  2. Callable InstructionProvider (Python function)")
    print()

    # --- Choose instruction approach ---
    print("Which approach?")
    print("  1. Jinja2 template (instruction_file + prompts_dir)  [default]")
    print("  2. Callable InstructionProvider (Python function)")
    approach = input("\nChoose (1-2): ").strip()
    agent = callable_agent if approach == "2" else template_agent  # <- Select which agent to use based on user choice. Both produce equivalent behavior, just implemented differently.
    approach_name = "callable" if approach == "2" else "template"

    # --- Choose meeting parameters ---
    print("\nWhat type of meeting is this?")
    print("  1. standup    2. brainstorm    3. review    4. planning    5. general")
    choice = input("Choose (1-5): ").strip()
    meeting_types = {"1": "standup", "2": "brainstorm", "3": "review", "4": "planning", "5": "general"}
    selected_type = meeting_types.get(choice, "general")

    formality = input("Formality (casual/formal) [casual]: ").strip().lower() or "casual"
    attendees_input = input("Attendees (comma-separated, or Enter to skip): ").strip()
    attendees = [a.strip() for a in attendees_input.split(",") if a.strip()] if attendees_input else []

    meeting_context = {  # <- Runtime context passed to runner.run_sync(). For the template agent, these become Jinja2 variables ({{ meeting_type }}, {{ formality }}, etc.). For the callable agent, these are passed to the function as context dict.
        "meeting_type": selected_type,
        "formality": formality,
        "attendees": attendees,
    }

    print(f"\nUsing: {approach_name} approach")
    print(f"Meeting: {selected_type} ({formality})")
    if attendees:
        print(f"Attendees: {', '.join(attendees)}")
    print("Start taking notes! Type 'summarize' for a summary, 'quit' to exit.\n")

    while True:
        user_input = input("[] > ")
        if user_input.strip().lower() in ("quit", "exit", "q"):
            print("Meeting ended. Goodbye!")
            break

        response = runner.run_sync(
            agent,
            user_message=user_input,
            context=meeting_context,  # <- Context flows to the instruction_file template (as Jinja2 variables) or to the callable (as function argument). Both mechanisms receive the same context dict.
        )
        print(f"[meeting-notes] > {response.final_text}\n")



"""
---
Tl;dr: This example demonstrates two approaches to dynamic agent instructions in AFK. The
template-based approach uses instruction_file="meeting_notes.md" and prompts_dir=Path("prompts/")
to load a Jinja2 template that renders with runtime context variables ({{ meeting_type }},
{% if formality == 'formal' %}, {{ attendees | join(', ') }}). The callable approach uses a
Python function that receives the context dict and returns a string. Both receive the same
runtime context from runner.run_sync(context={...}). Templates are best for text-heavy, static
prompt patterns; callables are best for logic-heavy, dynamic instruction generation.
---
---
What's next?
- Examine prompts/meeting_notes.md to see the full Jinja2 template with conditionals and filters.
- Try adding a new meeting type by editing the template — no Python changes needed!
- Set the AFK_AGENT_PROMPTS_DIR environment variable to load templates from a different directory.
- Experiment with async InstructionProviders (async def) for instructions that need I/O.
- Combine instruction_file with InstructionRole callbacks for layered instruction augmentation.
- Check out the Customer Support Router example to see InstructionRole in action!
---
"""
