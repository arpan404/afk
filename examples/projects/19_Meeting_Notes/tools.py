"""
---
name: Meeting Notes — Tools
description: Tool definitions for the meeting notes agent (add notes, action items, summaries).
tags: [tools]
---
---
This module defines all tools for the meeting notes agent. Tools are separated
into their own file to keep the main entry point focused on demonstrating
the instruction_file + prompts_dir pattern. Each tool operates on shared
in-memory state (notes and action items) that persists for the session.
---
"""

from pydantic import BaseModel, Field  # <- Pydantic for typed tool argument schemas.
from afk.tools import tool  # <- @tool decorator to create tools from plain functions.


# ===========================================================================
# Shared state for meeting notes and action items
# ===========================================================================

notes: list[dict] = []  # <- In-memory list of meeting notes.
action_items: list[dict] = []  # <- In-memory list of action items with assignee and description.
_note_counter: int = 0  # <- Auto-incrementing counter for note IDs.
_action_counter: int = 0  # <- Auto-incrementing counter for action item IDs.


# ===========================================================================
# Tool argument schemas
# ===========================================================================

class AddNoteArgs(BaseModel):
    content: str = Field(description="The content of the meeting note")


class AddActionItemArgs(BaseModel):
    description: str = Field(description="What needs to be done")
    assignee: str = Field(description="Who is responsible for this action item")


class EmptyArgs(BaseModel):
    pass


# ===========================================================================
# Tool definitions
# ===========================================================================

@tool(args_model=AddNoteArgs, name="add_note", description="Add a new note to the meeting record")
def add_note(args: AddNoteArgs) -> str:
    global _note_counter
    _note_counter += 1
    note = {"id": _note_counter, "content": args.content}
    notes.append(note)
    return f"Note #{_note_counter} added: {args.content}"


@tool(args_model=EmptyArgs, name="list_notes", description="List all notes taken during this meeting")
def list_notes(args: EmptyArgs) -> str:
    if not notes:
        return "No notes taken yet."
    lines = [f"  [{n['id']}] {n['content']}" for n in notes]
    return "Meeting Notes:\n" + "\n".join(lines)


@tool(args_model=EmptyArgs, name="summarize_notes", description="Generate a concise summary of all meeting notes")
def summarize_notes(args: EmptyArgs) -> str:
    if not notes:
        return "No notes to summarize."
    all_text = "\n".join(f"- {n['content']}" for n in notes)
    return f"Raw notes for summarization:\n{all_text}\n\nTotal notes: {len(notes)}"


@tool(args_model=AddActionItemArgs, name="add_action_item", description="Add an action item with an assignee")
def add_action_item(args: AddActionItemArgs) -> str:
    global _action_counter
    _action_counter += 1
    item = {"id": _action_counter, "description": args.description, "assignee": args.assignee, "done": False}
    action_items.append(item)
    return f"Action item #{_action_counter} added: '{args.description}' assigned to {args.assignee}"


@tool(args_model=EmptyArgs, name="list_action_items", description="List all action items from this meeting")
def list_action_items(args: EmptyArgs) -> str:
    if not action_items:
        return "No action items recorded yet."
    lines = []
    for item in action_items:
        status = "done" if item["done"] else "pending"
        lines.append(f"  [{item['id']}] [{status}] {item['description']} -> {item['assignee']}")
    return "Action Items:\n" + "\n".join(lines)
