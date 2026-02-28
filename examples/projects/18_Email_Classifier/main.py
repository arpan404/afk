"""
---
name: Email Classifier
description: An email classifier agent that analyzes emails and returns structured classification data.
tags: [agent, runner, tools, structured-output]
---
---
This example demonstrates how tools can return structured dictionary data (JSON-like) that the
agent interprets and presents to the user. The agent classifies emails by analyzing their subject,
sender, and body content, returning structured results with category, confidence score, and
reasoning. This pattern is essential for building agents that produce machine-readable output
alongside human-readable summaries.
---
"""

import json  # <- We use json.dumps to format structured tool output for readability.
from pydantic import BaseModel, Field  # <- Pydantic for typed tool argument schemas with validation.
from afk.core import Runner  # <- Runner executes agents and manages their state.
from afk.agents import Agent  # <- Agent defines the agent's model, instructions, and tools.
from afk.tools import tool  # <- @tool decorator to create tools from plain functions.


# ===========================================================================
# Simulated email inbox
# ===========================================================================

INBOX: list[dict] = [  # <- A simulated inbox with varied email types. In a real application, you would connect to an email API (Gmail, Outlook, etc.) — but static data keeps the focus on structured output patterns.
    {
        "id": 1,
        "from": "boss@company.com",
        "subject": "Q4 Budget Review — Action Required",
        "body": (
            "Hi team, please review the attached Q4 budget spreadsheet and submit your "
            "department estimates by Friday. We need accurate projections for the board meeting "
            "next week. Let me know if you have questions."
        ),
        "date": "2025-01-15 09:00",
    },
    {
        "id": 2,
        "from": "noreply@newsletter.dev",
        "subject": "This Week in Python: New PEP Proposals & Library Releases",
        "body": (
            "Welcome to your weekly Python digest! This week: PEP 750 introduces template strings, "
            "FastAPI 0.115 adds WebSocket improvements, and a roundup of the best new PyPI packages. "
            "Click here to read more. Unsubscribe at any time."
        ),
        "date": "2025-01-15 07:30",
    },
    {
        "id": 3,
        "from": "prince_ng@totallylegit.biz",
        "subject": "URGENT: You Have Won $5,000,000 — Claim Now!!!",
        "body": (
            "Dear beloved friend, I am Prince Okonkwo of Nigeria. You have been selected to receive "
            "$5,000,000 USD. Please send your bank details and social security number to claim your "
            "prize immediately. This offer expires in 24 hours!!!"
        ),
        "date": "2025-01-15 03:12",
    },
    {
        "id": 4,
        "from": "sarah.chen@company.com",
        "subject": "Lunch tomorrow?",
        "body": (
            "Hey! Want to grab lunch tomorrow at that new Thai place on Main Street? "
            "I heard they have amazing pad thai. Let me know if 12:30 works for you!"
        ),
        "date": "2025-01-14 17:45",
    },
    {
        "id": 5,
        "from": "github@notifications.github.com",
        "subject": "[afk-py] New pull request: Fix memory leak in SQLiteMemoryStore",
        "body": (
            "User @contributor123 opened a new pull request in arpan/afk-py. "
            "PR #342: Fix memory leak in SQLiteMemoryStore when connections are not properly closed. "
            "Changes: 3 files changed, 45 insertions(+), 12 deletions(-). "
            "Review requested from @arpan."
        ),
        "date": "2025-01-14 22:10",
    },
    {
        "id": 6,
        "from": "notifications@social.app",
        "subject": "Alex liked your photo",
        "body": (
            "Alex and 14 others liked your recent photo. You also have 3 new followers this week. "
            "Open the app to see your latest activity. Tap here to view."
        ),
        "date": "2025-01-14 16:00",
    },
]


# ===========================================================================
# Tool argument schemas
# ===========================================================================

class EmptyArgs(BaseModel):  # <- A schema with no fields — used for tools that take no input. The LLM still needs a schema to call the tool.
    pass


class EmailIdArgs(BaseModel):  # <- Schema for tools that operate on a specific email by ID.
    email_id: int = Field(description="The ID of the email to operate on")


class ClassifyEmailArgs(BaseModel):  # <- Schema for the classification tool. Takes an email ID to classify.
    email_id: int = Field(description="The ID of the email to classify")


# ===========================================================================
# Tool definitions
# ===========================================================================

@tool(args_model=EmptyArgs, name="list_inbox", description="List all emails in the inbox with their ID, sender, subject, and date")
def list_inbox(args: EmptyArgs) -> str:  # <- Returns a formatted list of all emails. The agent uses this to give the user an overview.
    lines = []
    for email in INBOX:
        lines.append(
            f"  [{email['id']}] From: {email['from']} | Subject: {email['subject']} | Date: {email['date']}"
        )
    return "Inbox:\n" + "\n".join(lines)


@tool(args_model=EmailIdArgs, name="get_email_details", description="Get the full details of a specific email including body content")
def get_email_details(args: EmailIdArgs) -> str:  # <- Returns the full email including body. Useful when the agent needs to inspect content before classifying.
    email = next((e for e in INBOX if e["id"] == args.email_id), None)
    if email is None:
        return f"Email with ID {args.email_id} not found. Valid IDs: {[e['id'] for e in INBOX]}"
    return (
        f"Email #{email['id']}:\n"
        f"  From: {email['from']}\n"
        f"  Subject: {email['subject']}\n"
        f"  Date: {email['date']}\n"
        f"  Body: {email['body']}"
    )


@tool(args_model=ClassifyEmailArgs, name="classify_email", description="Analyze an email and return a structured classification with category, confidence, and reasoning")
def classify_email(args: ClassifyEmailArgs) -> str:  # <- The core classification tool. Returns a structured JSON-like dict as a string. The agent interprets this structured output and presents it to the user.
    email = next((e for e in INBOX if e["id"] == args.email_id), None)
    if email is None:
        return json.dumps({"error": f"Email {args.email_id} not found"})

    # --- Simple heuristic classifier (in production, you'd use ML models) ---
    sender = email["from"].lower()
    subject = email["subject"].lower()
    body = email["body"].lower()
    text = f"{sender} {subject} {body}"  # <- Combine all fields for keyword matching.

    # --- Classification logic ---
    category = "general"  # <- Default category if no rules match.
    confidence = 0.5
    reasoning = "No strong signals detected."

    spam_signals = ["urgent", "claim now", "won", "bank details", "social security", "prince"]  # <- Common spam indicators.
    if sum(1 for s in spam_signals if s in text) >= 2:  # <- If multiple spam signals are present, classify as spam with high confidence.
        category = "spam"
        confidence = 0.95
        reasoning = "Multiple spam indicators: urgency language, request for personal information, suspicious sender."
    elif "action required" in text or sender.endswith("@company.com") and ("review" in text or "deadline" in text or "budget" in text):
        category = "work"
        confidence = 0.90
        reasoning = "Work-related content from company domain with action items."
    elif "unsubscribe" in text or "newsletter" in text or "digest" in text or "weekly" in text:
        category = "newsletter"
        confidence = 0.85
        reasoning = "Contains newsletter markers: digest format, unsubscribe link."
    elif "liked" in text or "followers" in text or "notifications@" in sender:
        category = "social"
        confidence = 0.80
        reasoning = "Social media notification from notification service."
    elif "pull request" in text or "github" in sender:
        category = "work"
        confidence = 0.85
        reasoning = "Development-related notification from GitHub."
    elif sender.endswith("@company.com"):
        category = "work"
        confidence = 0.70
        reasoning = "From company domain, likely work-related communication."

    result = {  # <- The structured classification result. Returning a dict (serialized as JSON) makes the output machine-readable while the agent can still explain it in natural language.
        "email_id": email["id"],
        "subject": email["subject"],
        "category": category,
        "confidence": round(confidence, 2),
        "reasoning": reasoning,
        "suggested_action": {  # <- Actionable suggestions based on the classification.
            "spam": "Move to spam folder and block sender",
            "work": "Flag as important and add to task list",
            "newsletter": "Archive or move to newsletters folder",
            "social": "Move to social folder",
            "general": "Keep in inbox for review",
        }.get(category, "Keep in inbox"),
    }

    return json.dumps(result, indent=2)  # <- Return as formatted JSON string. The agent sees this structured data and can extract specific fields to present to the user.


# ===========================================================================
# Agent and runner setup
# ===========================================================================

email_agent = Agent(
    name="email-classifier",  # <- What you want to call your agent.
    model="ollama_chat/gpt-oss:20b",  # <- The llm model the agent will use.
    instructions="""
    You are an email triage assistant. You help users manage their inbox by classifying emails
    into categories and suggesting actions.

    When the user asks to see their inbox:
    1. Use list_inbox to show all emails.

    When the user asks to classify an email:
    1. Use classify_email to get the structured classification.
    2. Present the results clearly: category, confidence percentage, reasoning, and suggested action.

    When the user asks to classify all emails:
    1. Classify each email one by one using classify_email.
    2. Present a summary table with all classifications.

    When the user wants details about a specific email:
    1. Use get_email_details to show the full content.

    Always present the structured classification data in a clear, readable format.
    Format confidence as a percentage (e.g., 95% instead of 0.95).

    **NOTE**: Be concise but thorough in your explanations!
    """,
    tools=[list_inbox, get_email_details, classify_email],
)

runner = Runner()

if __name__ == "__main__":
    print(
        "Email Classifier Agent (type 'quit' to exit)"
    )  # <- Welcome banner.
    print(
        "Try: 'show my inbox', 'classify email 3', or 'classify all emails'\n"
    )

    while True:  # <- Conversation loop for multi-turn interaction.
        user_input = input("[] > ")

        if user_input.strip().lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        response = runner.run_sync(
            email_agent, user_message=user_input
        )  # <- Run the agent synchronously. The agent will call tools as needed and return a natural language response that incorporates the structured data from tool outputs.

        print(
            f"[email-classifier] > {response.final_text}\n"
        )  # <- The agent's response includes structured classification data presented in a human-readable format.



"""
---
Tl;dr: This example creates an email classifier agent with tools that return structured JSON data (category,
confidence, reasoning, suggested_action). The classify_email tool uses heuristic rules to produce a
machine-readable dict, which the agent then interprets and presents to users in natural language. This
pattern is essential for building agents that bridge structured data and conversational interfaces.
---
---
What's next?
- Try adding a "batch_classify" tool that classifies all emails at once and returns a structured list.
- Experiment with having the agent output a markdown table summarizing all classifications.
- Replace the heuristic classifier with an actual ML model or LLM-based classification.
- Add a "move_email" tool that takes the classification's suggested_action and applies it.
- Explore using Pydantic models for the classification output itself (not just tool args) for stricter validation.
- Check out the other examples to see how to use streaming for real-time classification feedback!
---
"""
