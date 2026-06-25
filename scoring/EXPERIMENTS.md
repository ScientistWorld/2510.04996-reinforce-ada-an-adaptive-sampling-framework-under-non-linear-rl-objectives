# Experiment Overview

## synthetic_signal_recovery

Tests whether a policy can learn from binary verifier feedback on a prompt distribution where small response groups often contain no reward variation. The experiment measures held-out pass rate, initially hard and easy prompt slices, train-versus-eval gap, zero-variance group frequency, verifier-call cost, and policy-update group size to determine whether improved reward comes from better signal recovery rather than a larger training batch or overfitting.
