---
name: review
description: Grades the current build against the spec line by line, listing every gap and bug.
commands:
  - review
---

# /review Workflow

When this skill is triggered via `/review`, act as a strict, impartial QA reviewer. Compare the current codebase directly against the target specification document located in the `specs/` directory.

## Rules of Engagement

1. **Line-by-Line Audit**: Go through the specification requirement by requirement, checklist item by checklist item.
2. **Identify Failures**: List every single gap, bug, missing piece, or unhandled edge case. Explicitly name the exact requirement number or title from the spec that each issue violates.
3. **Draft Actionable Fixes**: If any item fails, do not write the code to fix it. Instead, document the exact structural or logical fixes required to resolve the issue. Hand these instructions back to the user so they can be processed by the `/build` skill.

## Evaluation Criteria

- **FAIL**: If even a single requirement, edge case, or definition of done item is missing or buggy, output your full list of required fixes and explicitly state that the build has **FAILED**.
- **PASS**: Only pass the build when every single requirement in the target specification is fully met, verified, and functioning.
