---
name: build
description: Reads the spec and builds exactly what it says with zero scope creep.
commands:
  - build
---

# /build Workflow

When this skill is triggered via `/build`, read the specification document located in `specs/<name>.md` and implement exactly what it describes. 

## Rules of Engagement
1. **Strict Scope Control**: Build only what is explicitly requested. Do not add speculative features, "nice-to-have" additions, or unrequested configurability.
2. **Surgical Coding**: Focus exclusively on the feature at hand. Do not refactor unrelated codebase files, rewrite adjacent functions, or clean up existing code unless the spec demands it.
3. **No Guessing**: If a requirement in the specification file is genuinely ambiguous or missing critical technical context, pause and ask the user for clarification before writing code.

## Output Summary
When you finish implementing the changes, provide a clean, bulleted checklist summarizing your work. Explicitly list which requirement numbers, edge cases, or "Definition of Done" items from the spec file you covered so the `/review` step can verify them line by line.
