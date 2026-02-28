"""
---
name: Document Approval — Tools
description: Document management tools for the approval workflow agent.
tags: [tools]
---
---
All tool definitions and simulated document storage for the document approval system.
---
"""

from pydantic import BaseModel, Field  # <- Pydantic for typed tool argument schemas.
from afk.tools import tool  # <- @tool decorator to create tools from plain functions.


# ===========================================================================
# Simulated document storage
# ===========================================================================

documents: dict[str, dict] = {}
_doc_counter: int = 0


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
def draft_document(args: DraftDocumentArgs) -> str:
    global _doc_counter
    _doc_counter += 1
    doc_id = f"DOC-{_doc_counter:03d}"
    documents[doc_id] = {
        "id": doc_id,
        "title": args.title,
        "content": args.content,
        "doc_type": args.doc_type,
        "status": "draft",
        "revisions": 0,
    }
    return (
        f"Document created: {doc_id}\n"
        f"  Title: {args.title}\n"
        f"  Type: {args.doc_type}\n"
        f"  Status: draft\n"
        f"  Content preview: {args.content[:100]}..."
    )


@tool(args_model=DocIdArgs, name="review_document", description="Review a document and provide feedback")
def review_document(args: DocIdArgs) -> str:
    doc = documents.get(args.doc_id)
    if doc is None:
        return f"Document {args.doc_id} not found."
    content = doc["content"]
    word_count = len(content.split())
    issues = []
    if not doc["title"]:
        issues.append("Missing title")
    if word_count < 10:
        issues.append(f"Content too short ({word_count} words, recommend 10+)")
    if issues:
        return f"Review of {args.doc_id} — Issues found:\n" + "\n".join(f"  - {i}" for i in issues)
    return f"Review of {args.doc_id} — Looks good!\n  Word count: {word_count}\n  Type: {doc['doc_type']}\n  Ready for finalization."


@tool(args_model=DocIdArgs, name="finalize_document", description="Finalize a document for publishing — sensitive, permanent action")
def finalize_document(args: DocIdArgs) -> str:
    doc = documents.get(args.doc_id)
    if doc is None:
        return f"Document {args.doc_id} not found."
    if doc["status"] == "finalized":
        return f"Document {args.doc_id} is already finalized."
    doc["status"] = "finalized"
    return (
        f"Document {args.doc_id} FINALIZED.\n"
        f"  Title: {doc['title']}\n"
        f"  Type: {doc['doc_type']}\n"
        f"  Status: finalized (permanent)"
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
