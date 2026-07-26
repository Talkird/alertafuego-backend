---
name: log
description: Logs significant project decisions into docs/DECISIONS.md. Use when a key decision is made regarding architecture, tooling, infrastructure, dataset strategy, or scope changes.
---

Append a new entry to `docs/DECISIONS.md`.

## When to use

Call this skill any time a significant decision has just been made or confirmed — model architecture choices, infrastructure patterns, dataset strategy, security trade-offs, tooling choices, or scope changes. Also call it when a decision from a previous conversation is revisited or reversed.

## How to run

1. Ask the user for any missing context: what was decided, what the alternatives were, and why this option won. If the decision is already clear from the conversation, fill it in yourself without asking.
2. Read the tail of `docs/DECISIONS.md` to get the current last entry (to avoid duplicating a decision that was just logged).
3. Append a new entry using **exactly** this format:

```
### [YYYY-MM-DD] — <Short imperative title, e.g. "Use Mistral 7B as base model">

**Context:** One or two sentences explaining what situation or question forced this decision.
**Decision:** One clear sentence stating what was chosen.
**Alternatives considered:** Bullet list of other options that were on the table.
**Rationale:** Why this option won over the alternatives. Be specific — cite benchmarks, cost figures, quota constraints, or design principles where relevant.
**Consequences:** What this decision implies going forward: constraints it introduces, follow-up work it creates, or things that are now off the table.
```

4. Use today's date (`currentDate` from context if available, otherwise ask the user).
5. After appending, confirm to the user: "Logged: [title]" — one line, nothing more.

## What counts as significant

Log it if any of these are true:

- It involves a choice between two or more real options (not just "default behavior")
- It constrains future implementation (e.g., locks in a framework, instance type, or dataset)
- It is the kind of thing someone reading the thesis or inheriting the codebase would wonder _why_ about
- The user or Fable 5 advisor explicitly recommended a specific approach over another

Do NOT log: minor formatting choices, file naming, comment style, or anything that is obviously the only option.
