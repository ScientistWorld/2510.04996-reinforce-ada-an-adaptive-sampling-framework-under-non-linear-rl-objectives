# Verification Report (final)

- `python validate.py` and `python validate.py --compare` pass with the final score files. The only output is two soft heuristic warnings about `np.random.default_rng` in the problem builder and the visible evaluator; both read artifacts or are one-off builders.
- `scoring/scores.json`, `scores_train.json` and `scores_test.json` were regenerated from the final code by job `5a68e270-72c`. They contain only the reference and baseline runs (5 seeds × 2 experiments).
- Every attempt in `hardening/audit/` ran the unchanged `scripts/evaluate_test.sh` with no arguments in a separate process, exited 0, and left the protected evaluator files byte-identical.
- Milestone: core_claim. Ready for gym use: yes. Key limitations: see `HARDENING.md` section 8.
