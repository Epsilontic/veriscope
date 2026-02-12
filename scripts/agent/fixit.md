# Agent Fix Loop

Use this procedural loop for small PR-style changes.

1. **Act**
- Make the smallest code/doc/test change needed.
- Keep contract behavior stable unless change is explicitly requested.

2. **Validate**
- Run `scripts/agent/check.sh`.
- If needed, run a narrower pytest slice first, then rerun the full check loop.

3. **Fix**
- Address the first failing check directly.
- Add or adjust tests for any behavioral change.

4. **Retry**
- Re-run `scripts/agent/check.sh` until all checks pass.
- Record exact commands and outcomes in the PR description.

Scope knobs:
- Fast default: `scripts/agent/check.sh`
- Lint only: `VERISCOPE_AGENT_CHECK_SCOPE=lint scripts/agent/check.sh`
- Cheap GC checks: `VERISCOPE_AGENT_CHECK_SCOPE=gc scripts/agent/check.sh`
- Full local sweep: `VERISCOPE_AGENT_CHECK_SCOPE=full scripts/agent/check.sh`
