"""
---
name: Document Approval
description: A document processing agent that uses InteractionProvider for human-in-the-loop approval before finalizing documents.
tags: [agent, runner, interaction, approval, human-in-the-loop]
---
---
This example demonstrates how to build agents that require human approval before performing
sensitive actions. Using AFK's InteractionProvider protocol and RunnerConfig's interaction_mode,
you can create workflows where the agent pauses, asks the user for approval, and only proceeds
if granted. This pattern is critical for production systems where certain actions (publishing,
deleting, spending money) must have a human checkpoint.
---
"""

import asyncio  # <- Async required for interaction provider.
from pydantic import BaseModel, Field  # <- Pydantic for typed tool argument schemas.
from afk.core import Runner, RunnerConfig  # <- Runner executes agents; RunnerConfig configures interaction mode.
from afk.agents import Agent  # <- Agent defines the agent's model, instructions, and tools.
from afk.tools import tool, ToolContext  # <- @tool decorator and ToolContext for execution context.


# ===========================================================================
# Simulated document storage
# ===========================================================================

documents: dict[str, dict] = {}  # <- In-memory document store. Maps doc_id to document data.
_doc_counter: int = 0  # <- Auto-incrementing document counter.


# ===========================================================================
# Tool argument schemas
# ===========================================================================

class DraftDocumentArgs(BaseModel):
    title: str = Field(description="Document title")
    content: str = Field(description="Document content/body text")
    doc_type: str = Field(description="Type of document: memo, report, proposal, letter")


class DocIdArgs(BaseModel):
    doc_id: str = Field(description="The document ID to operate on")


class EmptyArgs(BaseModel):
    pass


# ===========================================================================
# Tool definitions
# ===========================================================================

@tool(args_model=DraftDocumentArgs, name="draft_document", description="Create a new document draft")
def draft_document(args: DraftDocumentArgs) -> str:  # <- Creates a draft document. This is safe and doesn't require approval.
    global _doc_counter
    _doc_counter += 1
    doc_id = f"DOC-{_doc_counter:03d}"

    documents[doc_id] = {
        "id": doc_id,
        "title": args.title,
        "content": args.content,
        "doc_type": args.doc_type,
        "status": "draft",  # <- Documents start as drafts. They need review and approval before finalizing.
        "revisions": 0,
    }

    return (
        f"Document created: {doc_id}\n"
        f"  Title: {args.title}\n"
        f"  Type: {args.doc_type}\n"
        f"  Status: draft\n"
        f"  Content preview: {args.content[:100]}..."
    )


@tool(args_model=DocIdArgs, name="review_document", description="Review a document and provide feedback on its content")
def review_document(args: DocIdArgs) -> str:  # <- Reviews a document. Returns analysis but doesn't change status.
    doc = documents.get(args.doc_id)
    if doc is None:
        return f"Document {args.doc_id} not found."

    content = doc["content"]
    word_count = len(content.split())
    has_title = bool(doc["title"])
    is_long_enough = word_count >= 10

    issues = []
    if not has_title:
        issues.append("Missing title")
    if not is_long_enough:
        issues.append(f"Content too short ({word_count} words, recommend 10+)")

    if issues:
        return f"Review of {args.doc_id} — Issues found:\n" + "\n".join(f"  - {i}" for i in issues) + "\nStatus: needs revision"
    return f"Review of {args.doc_id} — Looks good!\n  Word count: {word_count}\n  Type: {doc['doc_type']}\n  Ready for finalization."


@tool(args_model=DocIdArgs, name="finalize_document", description="Finalize a document for publishing — this is a sensitive action that changes document status permanently")
def finalize_document(args: DocIdArgs) -> str:  # <- SENSITIVE action: finalizes a document. In a real system with InteractionProvider configured, the runner would pause for approval before executing this tool.
    doc = documents.get(args.doc_id)
    if doc is None:
        return f"Document {args.doc_id} not found."

    if doc["status"] == "finalized":
        return f"Document {args.doc_id} is already finalized."

    # --- This is where approval matters ---
    # When RunnerConfig has interaction_mode="interactive", the PolicyEngine or
    # policy_roles can trigger approval requests before this tool runs. For this
    # demo, we show the finalization proceeding (since we're running headless).
    doc["status"] = "finalized"  # <- In a production system, this would only happen after human approval via InteractionProvider.

    return (
        f"Document {args.doc_id} FINALIZED.\n"
        f"  Title: {doc['title']}\n"
        f"  Type: {doc['doc_type']}\n"
        f"  Status: finalized (permanent)\n\n"
        f"NOTE: In production with interaction_mode='interactive', this action would\n"
        f"pause and request human approval before proceeding."
    )


@tool(args_model=EmptyArgs, name="list_documents", description="List all documents with their current status")
def list_documents(args: EmptyArgs) -> str:
    if not documents:
        return "No documents yet. Use draft_document to create one."
    lines = ["Documents:"]
    for doc_id, doc in documents.items():
        lines.append(f"  [{doc_id}] {doc['title']} ({doc['doc_type']}) — {doc['status']}")
    return "\n".join(lines)


@tool(args_model=DocIdArgs, name="get_document", description="Get the full content of a document")
def get_document(args: DocIdArgs) -> str:
    doc = documents.get(args.doc_id)
    if doc is None:
        return f"Document {args.doc_id} not found."
    return (
        f"--- {doc['title']} ---\n"
        f"ID: {doc['id']} | Type: {doc['doc_type']} | Status: {doc['status']}\n"
        f"Revisions: {doc['revisions']}\n\n"
        f"{doc['content']}"
    )


# ===========================================================================
# Agent and runner setup
# ===========================================================================

# --- RunnerConfig demonstrates interaction settings ---
config = RunnerConfig(  # <- RunnerConfig controls how the runner handles interactions like approval requests.
    interaction_mode="headless",  # <- "headless" means no approval prompts (auto-decides). Set to "interactive" for human-in-the-loop. Options: "headless", "interactive", "external".
    approval_timeout_s=60.0,  # <- How long to wait for approval before timing out (when in interactive mode).
    approval_fallback="deny",  # <- What to do if approval times out: "allow" or "deny". "deny" is safer for sensitive operations.
    input_timeout_s=60.0,  # <- How long to wait for user input requests.
    input_fallback="deny",  # <- What to do if input times out.
    sanitize_tool_output=True,  # <- Sanitize tool output for safety.
    tool_output_max_chars=12_000,  # <- Maximum characters in tool output shown to the agent.
)

doc_agent = Agent(
    name="document-processor",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a document processing assistant. You help users create, review, and finalize documents.

    Workflow:
    1. Draft: Create new documents with draft_document.
    2. Review: Check documents for issues with review_document.
    3. Finalize: Mark documents as final with finalize_document (sensitive action!).

    Always review a document before finalizing it. Warn the user that finalization is
    permanent. If there are issues in the review, suggest fixes before finalizing.

    **IMPORTANT**: finalize_document is a sensitive action. In production with
    interaction_mode='interactive', this would require human approval. For this demo
    we're running in 'headless' mode, but the pattern is the same.
    """,
    tools=[draft_document, review_document, finalize_document, list_documents, get_document],
)

runner = Runner(config=config)  # <- Pass the RunnerConfig to the runner. The config controls interaction behavior.


if __name__ == "__main__":
    print("Document Approval Agent (type 'quit' to exit)")
    print("=" * 50)
    print("Interaction mode:", config.interaction_mode)
    print("Approval fallback:", config.approval_fallback)
    print()
    print("Try: 'Create a memo about the team offsite'")
    print("     'Review DOC-001'")
    print("     'Finalize DOC-001'\n")

    while True:
        user_input = input("[] > ")

        if user_input.strip().lower() in ("quit", "exit", "q"):
            finalized = sum(1 for d in documents.values() if d["status"] == "finalized")
            print(f"Session complete: {len(documents)} documents ({finalized} finalized). Goodbye!")
            break

        response = runner.run_sync(doc_agent, user_message=user_input)
        print(f"[document-processor] > {response.final_text}\n")



"""
---
Tl;dr: This example creates a document processing agent with a draft-review-finalize workflow,
configured with RunnerConfig for interaction behavior. The config demonstrates interaction_mode
("headless" vs "interactive"), approval_timeout_s, approval_fallback ("allow"/"deny"), and
tool output sanitization. In production with interaction_mode="interactive", sensitive actions
like finalize_document would pause and request human approval via an InteractionProvider before
proceeding. This pattern is critical for systems where certain actions require a human checkpoint.
---
---
What's next?
- Switch interaction_mode to "interactive" and implement a custom InteractionProvider to see approval prompts.
- Add a PolicyRole that triggers defer/request_approval for finalize_document specifically.
- Combine with PolicyEngine rules to automatically deny finalization for documents with review issues.
- Implement version history by tracking revisions in the document store.
- Add a "publish_document" tool with even stricter approval requirements.
- Check out the Content Moderator example for PolicyEngine-based tool gating!
---
"""
