# Reusable UR12e training workflow skill

> **Code style requirement: Economical code, exceptional readability, and excellent abstraction design.**

Date: 2026-09-30. Primary module TI03; related TI01/TI02/TI06.
Status: accepted for documentation delivery; scope explicitly authorized by the user on 2026-09-30.

## Aligned scope

Create a repository-local skill that reuses the successful 5,000-step Explorer
chain. SSH identities and scheduler accounts are configurable, including users
such as Jona or Sal. Require explicit hyperparameter alignment, a saved and
validated training TOML, and alignment of W&B identity/project/mode using the
user-selected Mac credential file. Preserve existing launch authorization;
do not introduce redundant approvals. Commit this skill to local main.

Deliver the discoverable skill, UI metadata, a self-contained historical chain
reference, this acceptance record, and a narrow AGENTS.md routing instruction.
Historical resources are examples, not required accounts or universal settings.
Do not submit jobs, connect to remote services, read live credentials, alter
training/inference implementation, publish images, push Git, or incorporate the
parent conversation's uncommitted changes as part of this documentation task.

## Implementation and acceptance

1. Read the local success evidence and current configuration/launcher contracts.
2. Document discovery, aligned TOML/W&B, artifact preparation, smoke, checkpoint
   roundtrip, bounded full training, monitoring, and final checkpoint verification.
3. Validate skill structure, links and portable use from the committed file set;
   check account independence, credential handling and resume boundaries.
4. Run applicable lightweight repository checks and commit only this delivery.

TI03-A02 extension: PASS for skill frontmatter/naming validation, Markdown link
resolution and manual workflow review. The review covered a new SSH user with a
different scheduler/W&B account, nonzero RTC with an incompatible prior norm
marker, already-authorized smoke-to-full promotion, offline logging without
online authentication, uncertain submission status, and zero-based checkpoint
naming. These are instruction reviews, not executed cloud training tests.

TI01-A02 extension: PASS for matching the historical reference to local completion
and launch evidence. Existing modified/untracked parent-task files were checked
by SHA256 and remained unchanged. The skill/reference has no required links to
those uncommitted artifacts or to another user's installed orchestration skill.

Validation: skill-creator quick_validate.py PASS using an isolated temporary
PyYAML 6.0.2 environment; local references and UI metadata checked; git diff
--check PASS. python -m training.check was attempted but could not execute its
formatter because the existing temporary environment lacks Black. The skill
validator initially lacked PyYAML and then passed in the isolated environment.
No Python implementation changed, so numerical/runtime suites were not rerun.
The main commit is limited to the skill's three files, AGENTS.md routing and
this plan; unrelated pending training changes are excluded.
GPU/remote execution: NOT RUN for this change; historical evidence is identified
as historical and does not validate future configurations or hardware behavior.
