# Experiment Overview

## synthetic_signal_recovery

Tests whether a policy can learn from binary verifier feedback on a prompt distribution where small response groups often contain no reward variation. The experiment measures held-out pass rate, initially hard and easy prompt slices, train-versus-eval gap, zero-variance group frequency, verifier-call cost, and policy-update group size to determine whether improved reward comes from better signal recovery rather than a larger training batch or overfitting.

## hard_signal_recovery

Repeats the same evaluation under a harder initial policy, making correct verifier hits rarer before training. This setting is intended to test whether gains are stronger when fixed small response groups lose more signal, matching the paper's hard-prompt analysis at a scaled level.
