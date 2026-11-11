This audit runs persisted-artifact methods through the unchanged hidden test.
Run `bash redteam/run_all.sh` **as a compute job**. Each named attempt also has
its own `run.sh`. Training, answer enumeration and fitting run only in jobs.

Every attempt temporarily stages `checkpoints/all_results`, starts a separate
`bash scripts/evaluate_test.sh` process with no arguments, saves its scores and
exit status in its folder, and restores the original data, results and scores.
No evaluator code, process, Python environment, or interpreter is modified.
`shortcut_replace_data` changes the visible problem generator, an input the
test imports. Its change is restored after the test. Model caches contain only
outputs from the reference's real training procedure and can be regenerated.

`analyze.py` is auditor-only reporting. Its copies of `reference.json` and the
original `scores_test.json` are never read by method preparation. Attempt code
uses only visible data, baseline and method interfaces; reconstruction shortcuts
do not use the private test split. A separate test process supplies all final
metrics. Failed tests are not credited with any preexisting score file.

Ranks depend on the future attempt cohort's metric standard deviations, which
cannot be known now. The report includes explicitly defined audit cohorts and
also identifies componentwise improvements that pay for every positive set of
normalizers. A missing scored experiment has exactly zero contribution under
the specified skip-missing rank. Negative coefficients are treated by magnitude.
