First audit job: 25 attempts, including 3 independently trained reference seeds,
two stronger-training variants, a ten-times-shorter variant, fixed-group
baselines, degenerate controls, and real persisted-artifact shortcuts. All
compute stays in the scheduled CPU job. No package installation is needed.

The runner restores all temporary artifact, data and scoring changes after
each trial, including expected evaluator failures. The root job entrypoint
will be restored from git after results are inspected.
