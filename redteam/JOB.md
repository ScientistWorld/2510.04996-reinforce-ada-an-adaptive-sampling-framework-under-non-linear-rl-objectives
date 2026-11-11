Audit complete: two scheduled CPU jobs, both exit code 0.

- `a282ce36-11e`: 25 attempts, including reference training on three seeds,
  genuine training variants, fixed-group baselines and 13 shortcut attempts.
- `d4f30614-e71`: two additional shortcut attempts, demonstrating a chance-level
  policy outranking the reference through negative costs and nearly perfect
  hidden-test accuracy from public generated evaluation answers.

All 27 attempts have persisted artifacts and status records. Twenty-five hidden
tests exited successfully, including one that emitted non-finite metrics. The
two intentional malformed-artifact failures produced no newly written scores;
the stale-file attempt is not credited with its forged score file.

`run_all.sh` now reproduces all 27 attempts in one job. Individual `run.sh` files
also work independently and regenerate reference outputs if needed. Normal
reference training is cached only to avoid repeating it for artifact shortcuts.
The payloads and evaluator run as separate processes with no arguments to the
hidden-test shell script. Evaluator hashes match across all attempts.

No reproduction milestone is upgraded by this audit. The original milestone
file is left intact because the audit may change only `redteam/` and
`REDTEAM.md`. The report records why the synthetic task demonstrates only part
of the paper's effect and why ranking remains unsuitable for scientific claims.
