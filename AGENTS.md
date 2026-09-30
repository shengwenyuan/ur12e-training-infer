# Project Instructions

- Conversation may follow the user; write plans and documentation in English.
- Use the stable TI module/acceptance IDs in meta_plan.md. Record aligned scope
  before implementation and actual PASS/FAIL/NOT RUN results afterward.
- Every plan must include: **Code style requirement: Economical code,
  exceptional readability, and excellent abstraction design.**
- Keep training/ and infer/ model-neutral at their shared boundaries. pi05_rtc
  is one backend. Preserve vendor numerical code and document narrow patches.
- Treat the 1011 training and UR12e exporter repositories as read-only sources.
  Do not inspect, import or depend on safeinfer.
- HOME is the collector's adopted [0,-90,-90,-90,90,0] degrees. Require measured
  arrival/settling before policy control. Space revokes pending actions and holds.
- Keep model/network work outside the control loop; preserve one actuator owner,
  fresh supervisor leases and controller watchdogs. Never automatically restart.
- Separate software, Docker, GPU, camera-shadow and physical acceptance. Software
  implementation permission does not authorize physical motion or cloud jobs.
- Use Google-style readable Python, explicit units/types and pinned dependencies.
  Run python -m training.check and relevant pytest cases. Keep data, weights,
  secrets, local station identities and /plans/ out of Git and image contexts.
- Docker Hub publication requires an explicit request; preparation/builds do not
  imply a push. Preserve colleagues' externally mounted data and checkpoints.

- For training configuration, W&B setup, remote launches or recovery, use the
  repository skill [ur12e-train](.agents/skills/ur12e-train/SKILL.md). Discover
  the selected SSH user, scheduler account and W&B entity independently; the
  historical Explorer run is evidence, not an account or parameter restriction.
