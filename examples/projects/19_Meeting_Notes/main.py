"""
---
name: Meeting Notes
description: A meeting notes agent that uses a dynamic InstructionProvider to adapt behavior based on meeting type and context.
tags: [agent, runner, tools, instruction-provider, context]
---
---
This example demonstrates how to use a callable InstructionProvider instead of a static instruction
string. The agent receives a function that generates instructions dynamically based on the runtime
context — so the same agent can behave differently for standups, brainstorms, reviews, or planning
sessions. This pattern is powerful for building adaptive agents that tailor their behavior to the
current situation without creating separate agent instances.
---
"""

from pydantic import BaseModel, Field  # <- Pydantic for typed tool argument schemas.
from afk.core import Runner  # <- Runner executes agents and manages their state.
from afk.agents import Agent  # <- Agent defines the agent's model, instructions, and tools.
from afk.tools import tool  # <- @tool decorator to create tools from plain functions.


# ===========================================================================
# Shared state for meeting notes and action items
# ===========================================================================

notes: list[dict] = []  # <- In-memory list of meeting notes. Each note is a dict with content and a timestamp-like index.
action_items: list[dict] = []  # <- In-memory list of action items with assignee and description.
_note_counter: int = 0  # <- Auto-incrementing counter for note IDs.
_action_counter: int = 0  # <- Auto-incrementing counter for action item IDs.


# ===========================================================================
# Tool argument schemas
# ===========================================================================

class AddNoteArgs(BaseModel):  # <- Schema for adding a new meeting note.
    content: str = Field(description="The content of the meeting note")


class AddActionItemArgs(BaseModel):  # <- Schema for adding an action item with an assignee.
    description: str = Field(description="What needs to be done")
    assignee: str = Field(description="Who is responsible for this action item")


class EmptyArgs(BaseModel):  # <- Schema with no fields for tools that need no input.
    pass


# ===========================================================================
# Tool definitions
# ===========================================================================

@tool(args_model=AddNoteArgs, name="add_note", description="Add a new note to the meeting record")
def add_note(args: AddNoteArgs) -> str:  # <- Appends a note to the shared notes list. Returns confirmation.
    global _note_counter
    _note_counter += 1
    note = {"id": _note_counter, "content": args.content}
    notes.append(note)
    return f"Note #{_note_counter} added: {args.content}"


@tool(args_model=EmptyArgs, name="list_notes", description="List all notes taken during this meeting")
def list_notes(args: EmptyArgs) -> str:  # <- Returns all notes taken so far, formatted with IDs.
    if not notes:
        return "No notes taken yet."
    lines = [f"  [{n['id']}] {n['content']}" for n in notes]
    return "Meeting Notes:\n" + "\n".join(lines)


@tool(args_model=EmptyArgs, name="summarize_notes", description="Generate a concise summary of all meeting notes")
def summarize_notes(args: EmptyArgs) -> str:  # <- Provides raw notes as context for the agent to summarize. The actual summarization happens via the LLM.
    if not notes:
        return "No notes to summarize."
    all_text = "\n".join(f"- {n['content']}" for n in notes)
    return f"Raw notes for summarization:\n{all_text}\n\nTotal notes: {len(notes)}"


@tool(args_model=AddActionItemArgs, name="add_action_item", description="Add an action item with an assignee")
def add_action_item(args: AddActionItemArgs) -> str:  # <- Tracks who needs to do what after the meeting.
    global _action_counter
    _action_counter += 1
    item = {"id": _action_counter, "description": args.description, "assignee": args.assignee, "done": False}
    action_items.append(item)
    return f"Action item #{_action_counter} added: '{args.description}' assigned to {args.assignee}"


@tool(args_model=EmptyArgs, name="list_action_items", description="List all action items from this meeting")
def list_action_items(args: EmptyArgs) -> str:  # <- Shows all action items with their assignees and completion status.
    if not action_items:
        return "No action items recorded yet."
    lines = []
    for item in action_items:
        status = "done" if item["done"] else "pending"
        lines.append(f"  [{item['id']}] [{status}] {item['description']} -> {item['assignee']}")
    return "Action Items:\n" + "\n".join(lines)


# ===========================================================================
# Dynamic instruction provider (the key concept)
# ===========================================================================

def meeting_instructions(context: dict) -> str:  # <- This is the InstructionProvider — a callable that receives the runtime context dict and returns an instruction string. The agent calls this function on every run, so the instructions adapt to the current context dynamically.
    meeting_type = context.get("meeting_type", "general")  # <- Extract the meeting type from context. Defaults to "general" if not set.
    formality = context.get("formality", "casual")  # <- Extract the formality level. Affects tone of the agent.
    attendees = context.get("attendees", [])  # <- Optional: list of people in the meeting.

    # --- Base instruction varies by meeting type ---
    type_instructions = {  # <- Each meeting type gets tailored instructions that guide the agent's focus and behavior.
        "standup": (
            "This is a daily standup meeting. Focus on three things per person:\n"
            "1. What did they do yesterday?\n"
            "2. What are they doing today?\n"
            "3. Any blockers?\n"
            "Keep notes brief and structured. Flag any blockers as action items immediately."
        ),
        "brainstorm": (
            "This is a brainstorming session. Your job is to capture ALL ideas without judgment.\n"
            "- Record every idea as a separate note, no matter how wild.\n"
            "- Group related ideas when asked.\n"
            "- Do NOT evaluate ideas — just capture them.\n"
            "- Encourage quantity over quality at this stage."
        ),
        "review": (
            "This is a review meeting (code review, design review, or sprint review).\n"
            "Focus on:\n"
            "- What was presented and by whom.\n"
            "- Feedback given (positive and constructive).\n"
            "- Decisions made.\n"
            "- Follow-up items and their owners.\n"
            "Be precise about who said what."
        ),
        "planning": (
            "This is a planning meeting. Focus on:\n"
            "- Goals and objectives being discussed.\n"
            "- Tasks identified and their estimated effort.\n"
            "- Dependencies between tasks.\n"
            "- Assignments and deadlines.\n"
            "Create action items for every task that gets assigned."
        ),
        "general": (
            "This is a general meeting. Take comprehensive notes covering:\n"
            "- Key discussion points.\n"
            "- Decisions made.\n"
            "- Action items and owners.\n"
            "Be thorough but concise."
        ),
    }

    base = type_instructions.get(meeting_type, type_instructions["general"])  # <- Fall back to general if unknown type.

    # --- Adjust tone based on formality ---
    tone = (  # <- Dynamic tone adjustment based on the formality context key.
        "Use a professional, formal tone. Avoid casual language."
        if formality == "formal"
        else "Use a friendly, conversational tone. Keep things light and approachable."
    )

    # --- Build attendee awareness ---
    attendee_note = ""
    if attendees:
        names = ", ".join(attendees)
        attendee_note = f"\nAttendees in this meeting: {names}. Reference them by name when possible."

    return (  # <- Assemble the final instruction string from all the dynamic parts.
        f"You are a meeting notes assistant.\n\n"
        f"Meeting type: {meeting_type}\n\n"
        f"{base}\n\n"
        f"{tone}\n"
        f"{attendee_note}\n\n"
        f"Use the available tools to record notes and action items. "
        f"When the user says 'summarize' or 'wrap up', provide a complete meeting summary."
    )


# ===========================================================================
# Agent and runner setup
# ===========================================================================

notes_agent = Agent(
    name="meeting-notes",  # <- What you want to call your agent.
    model="ollama_chat/gpt-oss:20b",  # <- The llm model the agent will use.
    instructions=meeting_instructions,  # <- Instead of a static string, we pass the callable. The runner calls this function with the current context before each run, so the agent's instructions adapt dynamically.
    context_defaults={  # <- Default context values. These are used if no overrides are provided at runtime. The instruction provider reads these keys.
        "meeting_type": "general",
        "formality": "casual",
        "attendees": [],
    },
    tools=[add_note, list_notes, summarize_notes, add_action_item, list_action_items],
)

runner = Runner()


if __name__ == "__main__":
    print("Meeting Notes Agent (type 'quit' to exit)")
    print("=" * 45)

    # --- Let the user choose a meeting type ---
    print("\nWhat type of meeting is this?")
    print("  1. standup")
    print("  2. brainstorm")
    print("  3. review")
    print("  4. planning")
    print("  5. general")

    choice = input("\nChoose (1-5): ").strip()  # <- The user picks a meeting type, which gets passed as context to the agent.
    meeting_types = {"1": "standup", "2": "brainstorm", "3": "review", "4": "planning", "5": "general"}
    selected_type = meeting_types.get(choice, "general")

    formality = input("Formality (casual/formal) [casual]: ").strip().lower() or "casual"  # <- The user can also set the formality level.

    attendees_input = input("Attendees (comma-separated, or press Enter to skip): ").strip()  # <- Optional attendee list.
    attendees = [a.strip() for a in attendees_input.split(",") if a.strip()] if attendees_input else []

    # --- Build the runtime context ---
    meeting_context = {  # <- This context dict is passed to the runner and forwarded to the instruction provider. It overrides the context_defaults set on the agent.
        "meeting_type": selected_type,
        "formality": formality,
        "attendees": attendees,
    }

    print(f"\nStarting {selected_type} meeting ({formality} tone)")
    if attendees:
        print(f"Attendees: {', '.join(attendees)}")
    print("Start taking notes! Type 'summarize' to get a summary, 'quit' to exit.\n")

    while True:  # <- Conversation loop for the meeting.
        user_input = input("[] > ")

        if user_input.strip().lower() in ("quit", "exit", "q"):
            print("Meeting ended. Goodbye!")
            break

        response = runner.run_sync(
            notes_agent,
            user_message=user_input,
            context=meeting_context,  # <- Pass the runtime context here. The instruction provider receives this context and generates appropriate instructions for this specific meeting type. This is how you override context_defaults at runtime.
        )

        print(f"[meeting-notes] > {response.final_text}\n")



"""
---
Tl;dr: This example creates a meeting notes agent with a dynamic InstructionProvider — a callable that
generates instructions based on runtime context (meeting_type, formality, attendees). Instead of a static
instruction string, the agent's instructions= parameter receives a function that adapts behavior per
request. The same agent handles standups, brainstorms, reviews, and planning sessions with different
note-taking strategies. Context is provided via context_defaults on the agent and overridden at runtime
via runner.run_sync(context={...}).
---
---
What's next?
- Try switching meeting types between runs to see how the agent's behavior changes.
- Experiment with async instruction providers (async def) for instructions that require I/O (e.g. fetching a template from a database).
- Add a "meeting_language" context key to make the instruction provider generate instructions in different languages.
- Combine the dynamic instruction provider with subagents — each subagent could have its own instruction provider.
- Explore using instruction_file and prompts_dir for template-based instructions loaded from disk.
- Check out the other examples to see how context_defaults and inherit_context_keys work with subagent delegation!
---
"""
