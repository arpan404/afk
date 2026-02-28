---
name: task-ops
description: Task management best practices and operational guidelines for the production task manager agent. Provides context on prioritization strategies, workflow patterns, and team coordination.
---

# Task Operations Skill

Use this skill when managing tasks, planning work, or providing productivity advice.

## Task Prioritization Guidelines

When helping users prioritize tasks, follow this framework:

1. **Critical**: Production issues, security vulnerabilities, data loss risks. Handle immediately.
2. **High**: Feature deadlines within 48 hours, blocking dependencies, customer-facing bugs.
3. **Medium**: Planned feature work, non-blocking bugs, documentation updates.
4. **Low**: Nice-to-haves, refactoring, exploratory research.

## Category Definitions

- **bug**: Defects in existing functionality. Always include reproduction steps.
- **feature**: New capabilities or enhancements. Should reference requirements.
- **docs**: Documentation tasks. Include target audience (user, developer, ops).
- **ops**: Infrastructure, deployment, monitoring. Include environment context.
- **general**: Uncategorized work. Encourage users to pick a specific category.

## Workflow Patterns

### Daily Standup Review
When asked to summarize or review tasks:
- List critical/high priority open tasks first
- Highlight tasks that have been open for more than 3 days
- Suggest tasks that could be completed quickly (low-hanging fruit)

### Sprint Planning
When asked to plan or prioritize:
- Group tasks by category
- Identify dependencies between tasks
- Recommend a balanced mix of bug fixes and feature work
- Flag if any single category has too many open items (>5)

### Overload Detection
If a user has more than 10 open tasks:
- Warn them about cognitive overload
- Suggest breaking large tasks into smaller ones
- Recommend completing or closing stale tasks

## Delegation Advice

When delegating to specialist subagents:
- Use the **analyst** for retrospective analysis (what happened, trends, bottlenecks)
- Use the **planner** for forward-looking strategy (what to do next, scheduling)
- Handle CRUD operations directly — don't delegate simple creates/updates/deletes
