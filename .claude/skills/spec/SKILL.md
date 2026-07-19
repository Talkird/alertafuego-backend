---
name: spec
description: Interviews you until it understands what you want, then writes a detailed spec.
commands:
  - spec
---

# /spec Workflow

When this skill is triggered via `/spec`, interview the user about the feature or app they want to build.

## Rules of Engagement

1. Ask exactly one focused question at a time.
2. Continue interviewing until you fully understand the core goal, must-have requirements, technological constraints, and success metrics.
3. Do not start building any code or application files during this step.

## Output Structure

When you have gathered enough information, synthesize the details into a clear, detailed specification document. Save this document directly to a new file at `specs/<name>.md`.

The generated specification file must include:

- **Objective**: A high-level summary of the feature or app.
- **Requirements**: Clear, atomic, and actionable functional requirements.
- **Edge Cases**: Tricky scenarios or error states that must be handled.
- **Definition of Done**: A concrete checklist that someone can test the final build against.
