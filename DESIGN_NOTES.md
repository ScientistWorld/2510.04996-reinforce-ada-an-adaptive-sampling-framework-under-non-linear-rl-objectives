# Design Notes (final gym)

- **Problem.** `data/seqverify.py` holds the public interface: policy class, verifier, protocol constants and artifact writer. The committed instance is `data/seqverify/*.json`, built once by `eval/test/build_problem.py` from a secret seed. The builder lives in `eval/test/` because it can regenerate the hidden answers.
- **Hidden test.** `eval/test/evaluate.py` is self-contained: it imports neither `data/` nor `method/`. It holds 4000 hidden prompts with plaintext answers and fixed hard/easy slices (by base-policy pass rate).
  - Scored metrics are greedy accuracies. Temperature-1 `pass1` is a diagnostic.
  - It requires 5 distinct runs per method and experiment. Failed, missing and duplicate runs are worst values (0), and every method on disk is scored.
- **Visible split.** `eval/train/evaluate.py` uses its own code and 1000 visible prompts with hashed answers. Greedy responses are verified exactly; `pass1` is Ave@32.
- **Canonical scores.** `eval/evaluate_results.py` adds unscored self-reported cost diagnostics from run histories to `scoring/scores.json`.
- **Protocol.** `data.seqverify.PROTOCOL` fixes 600 updates × 256 prompts, ≤ 32 samples and 4 update responses per prompt per update, Adam lr 0.1 with 10-step warm-up and entropy bonus 1e-4, and 5 runs. It is identical for every method; the shared batch size and learning rate were calibrated once (`hardening/calibration/`).
- **Audit.** `hardening/` holds the audit harness, the variants, the calibration and the analysis behind `HARDENING.md`.
