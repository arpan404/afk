"""
---
name: Content Moderator
description: A content moderation agent that uses PolicyEngine with PolicyRule to gate tool calls based on content patterns.
tags: [agent, runner, tools, policy-engine, policy-rule, moderation]
---
---
This example demonstrates AFK's PolicyEngine and PolicyRule system for deterministic tool gating.
Instead of relying on the LLM to decide whether a tool should run, you define declarative rules
that the runner evaluates BEFORE executing any tool call. Each PolicyRule specifies a rule_id,
an action ("allow", "deny", or "defer"), a priority, and conditions (like which tool name to
match). The engine evaluates all matching rules in priority order and the highest-priority match
wins. This pattern is essential for safety-critical workflows where you need hard guarantees --
you can block dangerous tool calls, require approval for sensitive operations, or allow routine
actions unconditionally. The content moderator agent decides whether user-submitted posts should
be published, flagged for review, or rejected outright, with the PolicyEngine enforcing guardrails
around the publish_post tool.
---
"""

from pydantic import BaseModel, Field  # <- Pydantic for typed tool argument schemas.
from afk.core import Runner  # <- Runner orchestrates agent execution and applies policy checks before tool calls.
from afk.agents import Agent  # <- Agent defines the agent's model, instructions, and tools.
from afk.tools import tool  # <- @tool decorator to create tools from plain functions.
from afk.agents.policy.engine import (  # <- The policy subsystem for declarative tool gating.
    PolicyEngine,  # <- The engine that evaluates rules against incoming events. It checks all rules, picks the highest-priority match, and returns a PolicyDecision ("allow", "deny", or "defer").
    PolicyRule,  # <- A single rule: rule_id, action, priority, enabled, subjects, reason. Higher priority wins. Subjects filter which event types the rule applies to (e.g., "tool_call", "subagent_call", "any").
    PolicyRuleCondition,  # <- Conditions for matching: tool_name, tool_name_pattern, context_equals, context_has_keys. A rule only fires when ALL conditions match the incoming event.
)


# ===========================================================================
# Simulated content database
# ===========================================================================

CONTENT_DB: list[dict] = []  # <- Stores all posts that have been processed. In a real system this would be a database — but an in-memory list keeps the focus on policy gating.

FLAGGED_CONTENT: list[dict] = []  # <- Posts the agent flagged for human review.

REJECTED_CONTENT: list[dict] = []  # <- Posts the agent rejected outright.

PUBLISHED_CONTENT: list[dict] = []  # <- Posts that were successfully published.


# ===========================================================================
# Sensitive patterns — the policy engine uses these to gate tool calls
# ===========================================================================

SENSITIVE_KEYWORDS: list[str] = [  # <- Keywords that make content "sensitive." Posts containing these are flagged or denied.
    "violence", "hate", "harassment", "threat", "spam",
    "scam", "phishing", "explicit", "dangerous", "illegal",
]

FLAGGABLE_KEYWORDS: list[str] = [  # <- Keywords that trigger manual review. Less severe than rejection, but still require a human check.
    "controversial", "political", "medical advice", "financial advice",
    "gambling", "supplement", "miracle", "cure", "guaranteed",
]


# ===========================================================================
# Tool argument schemas
# ===========================================================================

class AnalyzeContentArgs(BaseModel):  # <- Schema for the content analysis tool.
    content: str = Field(description="The text content of the post to analyze")
    author: str = Field(description="The username of the post author")


class PublishPostArgs(BaseModel):  # <- Schema for the publish tool. This is the tool that the PolicyEngine gates.
    content: str = Field(description="The content to publish")
    author: str = Field(description="The author of the post")
    category: str = Field(description="Content category: general, news, opinion, creative")


class FlagContentArgs(BaseModel):  # <- Schema for flagging content for human review.
    content: str = Field(description="The content being flagged")
    author: str = Field(description="The author of the post")
    reason: str = Field(description="Why this content is being flagged for review")


class RejectContentArgs(BaseModel):  # <- Schema for rejecting content.
    content: str = Field(description="The content being rejected")
    author: str = Field(description="The author of the post")
    reason: str = Field(description="Why this content is being rejected")


# ===========================================================================
# Tool definitions
# ===========================================================================

@tool(  # <- The analysis tool is always allowed — it just inspects content. No policy gating needed.
    args_model=AnalyzeContentArgs,
    name="analyze_content",
    description="Analyze a post's content for policy violations, tone, and category. Returns a detailed analysis report.",
)
def analyze_content(args: AnalyzeContentArgs) -> str:
    content_lower = args.content.lower()  # <- Case-insensitive keyword matching.

    # --- Check for hard-reject keywords ---
    found_sensitive = [kw for kw in SENSITIVE_KEYWORDS if kw in content_lower]  # <- Scan the content for keywords that trigger outright rejection.

    # --- Check for flaggable keywords ---
    found_flaggable = [kw for kw in FLAGGABLE_KEYWORDS if kw in content_lower]  # <- Scan for keywords that trigger manual review.

    # --- Basic content metrics ---
    word_count = len(args.content.split())
    char_count = len(args.content)
    has_urls = "http://" in content_lower or "https://" in content_lower  # <- URLs can indicate spam or phishing.
    all_caps_ratio = sum(1 for c in args.content if c.isupper()) / max(char_count, 1)  # <- High caps ratio often means shouting or spam.

    # --- Build analysis report ---
    status = "clean"
    if found_sensitive:
        status = "reject"  # <- Hard violations → immediate rejection.
    elif found_flaggable:
        status = "flag"  # <- Soft violations → flag for review.
    elif all_caps_ratio > 0.7 and word_count > 5:
        status = "flag"  # <- Excessive caps is suspicious.
    elif has_urls and word_count < 10:
        status = "flag"  # <- Short posts with URLs are often spam.

    report_lines = [
        f"Content Analysis Report",
        f"{'=' * 40}",
        f"Author: {args.author}",
        f"Word count: {word_count}",
        f"Character count: {char_count}",
        f"Contains URLs: {'yes' if has_urls else 'no'}",
        f"ALL-CAPS ratio: {all_caps_ratio:.0%}",
        f"Sensitive keywords found: {', '.join(found_sensitive) if found_sensitive else 'none'}",
        f"Flaggable keywords found: {', '.join(found_flaggable) if found_flaggable else 'none'}",
        f"",
        f"Recommendation: {status.upper()}",
    ]

    if status == "reject":
        report_lines.append(f"Reason: Contains policy-violating content ({', '.join(found_sensitive)})")
    elif status == "flag":
        reasons = []
        if found_flaggable:
            reasons.append(f"sensitive topics ({', '.join(found_flaggable)})")
        if all_caps_ratio > 0.7:
            reasons.append("excessive capitalization")
        if has_urls and word_count < 10:
            reasons.append("short post with URLs (possible spam)")
        report_lines.append(f"Reason: Requires review — {', '.join(reasons)}")
    else:
        report_lines.append("Reason: Content appears safe for publication")

    return "\n".join(report_lines)


@tool(  # <- The publish tool IS gated by the PolicyEngine. The engine's rules can DENY this tool call when the context contains flagged content.
    args_model=PublishPostArgs,
    name="publish_post",
    description="Publish a post to the content feed. This tool is gated by the PolicyEngine — it may be denied if the content violates policies.",
)
def publish_post(args: PublishPostArgs) -> str:
    post = {
        "content": args.content,
        "author": args.author,
        "category": args.category,
        "status": "published",
    }
    PUBLISHED_CONTENT.append(post)  # <- Add to the published feed.
    CONTENT_DB.append(post)  # <- Also record in the global DB.
    return (
        f"Post published successfully!\n"
        f"Author: {args.author}\n"
        f"Category: {args.category}\n"
        f"Content preview: {args.content[:100]}{'...' if len(args.content) > 100 else ''}\n"
        f"Total published posts: {len(PUBLISHED_CONTENT)}"
    )


@tool(  # <- Flagging is always allowed — it's a safe operation.
    args_model=FlagContentArgs,
    name="flag_content",
    description="Flag content for human moderator review. Use when content is borderline or needs a second opinion.",
)
def flag_content(args: FlagContentArgs) -> str:
    entry = {
        "content": args.content,
        "author": args.author,
        "reason": args.reason,
        "status": "flagged",
    }
    FLAGGED_CONTENT.append(entry)
    CONTENT_DB.append(entry)
    return (
        f"Content flagged for review.\n"
        f"Author: {args.author}\n"
        f"Reason: {args.reason}\n"
        f"Flagged posts in queue: {len(FLAGGED_CONTENT)}"
    )


@tool(  # <- Rejection is always allowed — it's a protective action.
    args_model=RejectContentArgs,
    name="reject_content",
    description="Reject content that clearly violates content policies. Use for severe violations.",
)
def reject_content(args: RejectContentArgs) -> str:
    entry = {
        "content": args.content,
        "author": args.author,
        "reason": args.reason,
        "status": "rejected",
    }
    REJECTED_CONTENT.append(entry)
    CONTENT_DB.append(entry)
    return (
        f"Content rejected.\n"
        f"Author: {args.author}\n"
        f"Reason: {args.reason}\n"
        f"Total rejected posts: {len(REJECTED_CONTENT)}"
    )


# ===========================================================================
# PolicyEngine — declarative rules for tool gating
# ===========================================================================

policy_engine = PolicyEngine(  # <- The PolicyEngine evaluates rules deterministically. It does NOT use AI — it's pure Python logic. Rules are sorted by priority descending, and the first match wins.
    rules=[
        PolicyRule(  # <- Rule 1: DENY publish_post when context signals content is flagged. This prevents the agent from publishing borderline content even if the LLM decides to.
            rule_id="deny-publish-flagged",
            action="deny",  # <- "deny" means the tool call is blocked and an error is returned to the agent. Other actions: "allow" (let it through), "defer" (pause for external approval).
            priority=200,  # <- Higher priority = evaluated first. 200 beats the default allow of 50.
            enabled=True,  # <- Set to False to temporarily disable a rule without removing it.
            subjects=["tool_call"],  # <- This rule only applies to tool_call events (not subagent_call or llm_call).
            reason="Content has been flagged for review — publishing is blocked until a human moderator approves it.",  # <- Reason is returned to the agent so it can explain why the action was denied.
            condition=PolicyRuleCondition(  # <- Conditions narrow when this rule fires. Here: only when the tool being called is "publish_post" AND the context has content_status="flagged".
                tool_name="publish_post",  # <- Match only the publish_post tool. Other tools (analyze, flag, reject) are unaffected.
                context_equals={"content_status": "flagged"},  # <- The context must contain this key-value pair for the rule to match. We set this in the run context when calling the runner.
            ),
        ),
        PolicyRule(  # <- Rule 2: DENY publish_post when content was rejected. Double safety: even if the LLM somehow tries to publish rejected content, this rule stops it.
            rule_id="deny-publish-rejected",
            action="deny",
            priority=300,  # <- Highest priority for rejected content — overrides everything.
            enabled=True,
            subjects=["tool_call"],
            reason="Content has been rejected — publishing is permanently blocked for this post.",
            condition=PolicyRuleCondition(
                tool_name="publish_post",
                context_equals={"content_status": "rejected"},  # <- Context must signal that content was already rejected.
            ),
        ),
        PolicyRule(  # <- Rule 3: DEFER publish_post for sensitive categories. "defer" means the tool call pauses and can be resumed after external approval.
            rule_id="defer-publish-sensitive-category",
            action="defer",  # <- "defer" puts the tool call on hold. In a production system, a human approver would review and either approve or deny.
            priority=150,  # <- Lower than deny rules, higher than the default allow.
            enabled=True,
            subjects=["tool_call"],
            reason="Posts in sensitive categories require moderator approval before publishing.",
            condition=PolicyRuleCondition(
                tool_name="publish_post",
                context_equals={"content_category": "opinion"},  # <- Opinion pieces are sensitive — they need human review before publishing.
            ),
        ),
        PolicyRule(  # <- Rule 4: Default ALLOW for all tool calls. This is the fallback — if no deny/defer rule matches, allow the call.
            rule_id="default-allow",
            action="allow",  # <- Explicit default. If you remove this, the engine's built-in default is also "allow", but being explicit is better for auditing.
            priority=50,  # <- Low priority so any specific deny/defer rule overrides it.
            enabled=True,
            subjects=["any"],  # <- Applies to all event types: tool_call, subagent_call, llm_call.
            reason="Default policy: allow all actions unless a higher-priority rule blocks them.",
            condition=PolicyRuleCondition(),  # <- Empty condition matches everything.
        ),
    ],
)

"""
Policy rule evaluation order (sorted by priority descending):

  Priority 300: deny-publish-rejected     — blocks publish for rejected content
  Priority 200: deny-publish-flagged      — blocks publish for flagged content
  Priority 150: defer-publish-sensitive    — defers publish for opinion category
  Priority  50: default-allow             — allows everything else

When the agent tries to call publish_post, the runner checks these rules.
If content_status="rejected" in context → denied (priority 300 wins).
If content_status="flagged" in context → denied (priority 200 wins).
If content_category="opinion" in context → deferred (priority 150 wins).
Otherwise → allowed (priority 50 default).
"""


# ===========================================================================
# Agent setup — PolicyEngine is passed to the Agent constructor
# ===========================================================================

moderator = Agent(
    name="content-moderator",  # <- The agent's display name.
    model="ollama_chat/gpt-oss:20b",  # <- The LLM model the agent will use.
    instructions="""
    You are a content moderation agent. Your job is to review user-submitted posts and decide
    whether they should be published, flagged for review, or rejected.

    Workflow for each post:
    1. ALWAYS analyze the content first using the analyze_content tool.
    2. Based on the analysis:
       - If the content is CLEAN: publish it using publish_post.
       - If the content is BORDERLINE: flag it for human review using flag_content.
       - If the content VIOLATES policies: reject it using reject_content.
    3. Explain your decision clearly to the user.

    Important: The system has a PolicyEngine that may DENY your publish_post calls if the
    content has been flagged or rejected. If a publish attempt is denied, explain to the user
    why and suggest they modify their content.

    Be fair, consistent, and transparent about moderation decisions.
    """,  # <- Instructions tell the agent HOW to moderate. The PolicyEngine adds HARD guardrails on top of the LLM's judgment.
    tools=[analyze_content, publish_post, flag_content, reject_content],  # <- All four tools are available. The PolicyEngine selectively gates publish_post.
    policy_engine=policy_engine,  # <- Attach the PolicyEngine to the agent. The Runner evaluates its rules before every tool call. This is the KEY line — without it, the rules are never checked.
)

runner = Runner()  # <- A single Runner instance handles all agent executions.


# ===========================================================================
# Sample posts for demonstration
# ===========================================================================

SAMPLE_POSTS: list[dict] = [  # <- Pre-built sample posts for easy testing. Each has different moderation outcomes.
    {
        "author": "alice",
        "content": "Just had the most amazing sunset hike today! The trail through the mountains was breathtaking and the wildflowers were in full bloom.",
        "expected": "PUBLISH (clean content)",
    },
    {
        "author": "bob",
        "content": "This GUARANTEED miracle cure will solve all your problems! Visit http://totallynotascam.com for more info. Act NOW!",
        "expected": "FLAG (spam keywords, URL, promotional language)",
    },
    {
        "author": "charlie",
        "content": "I think this new policy is controversial but we should have an open political debate about it.",
        "expected": "FLAG (political/controversial keywords)",
    },
    {
        "author": "diana",
        "content": "Stop the violence and hate! We need to end harassment and threats in online spaces.",
        "expected": "REJECT (contains sensitive keywords even in positive context)",
    },
]


# ===========================================================================
# Main entry point — interactive conversation loop
# ===========================================================================

if __name__ == "__main__":
    print("Content Moderator Agent (type 'quit' to exit)")
    print("=" * 50)
    print("Submit posts for moderation. The agent will analyze, then publish/flag/reject.")
    print("\nSample posts to try:")
    for i, post in enumerate(SAMPLE_POSTS, 1):
        print(f"  {i}. [{post['author']}] {post['content'][:60]}...")
        print(f"     Expected: {post['expected']}")
    print(f"\nOr type your own post to moderate.\n")

    while True:  # <- Conversation loop for the moderation interaction.
        user_input = input("[] > ")

        if user_input.strip().lower() in ("quit", "exit", "q"):
            print(f"\nModeration summary:")
            print(f"  Published: {len(PUBLISHED_CONTENT)}")
            print(f"  Flagged:   {len(FLAGGED_CONTENT)}")
            print(f"  Rejected:  {len(REJECTED_CONTENT)}")
            print("Goodbye!")
            break

        response = runner.run_sync(  # <- Synchronous run — blocks until the agent finishes. Internally, the Runner checks the PolicyEngine before every tool call.
            moderator,
            user_message=user_input,
        )

        print(f"[moderator] > {response.final_text}\n")



"""
---
Tl;dr: This example creates a content moderation agent with a PolicyEngine that gates the
publish_post tool using declarative PolicyRules. Four rules are defined: deny publishing for
rejected content (priority 300), deny publishing for flagged content (priority 200), defer
publishing for opinion-category posts (priority 150), and a default allow-all fallback (priority
50). The engine evaluates rules by priority — the highest-priority matching rule wins. Rules use
PolicyRuleCondition to match on tool_name and context_equals, so the decision is deterministic,
not AI-based. The agent has four tools (analyze_content, publish_post, flag_content, reject_content)
and the PolicyEngine only gates publish_post. This pattern is essential for safety-critical workflows
where hard guardrails must override the LLM's judgment.
---
---
What's next?
- Try adding a context key to the runner call (e.g., context={"content_status": "flagged"}) to see the deny rule block publishing.
- Experiment with the "defer" action by setting content_category="opinion" in context — this simulates queuing for human approval.
- Add a new PolicyRule that denies ALL tool calls for a specific author (e.g., a banned user) using context_equals={"author": "banned_user"}.
- Use tool_name_pattern="publish_*" to gate multiple publish-related tools with a single rule.
- Combine the PolicyEngine with a SubagentRouter to build a moderation pipeline where different agents handle different severity levels.
- Explore the PolicyEngine with policy_roles for dynamic, callback-based policy decisions alongside static rules!
---
"""
