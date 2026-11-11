# Experiment Overview

Both experiments train a fixed policy class from binary verifier feedback under the same fixed training budget. Both score the trained policy on one set of hidden evaluation prompts drawn from the full prompt distribution.

## seqverify_main

Establishes whether a method learns more from a limited sampling and update budget when the training prompts are of moderate difficulty for the starting policy: the standard curated training regime. Scored by held-out greedy accuracy overall, on prompts that start out hardest, and on prompts that start out easiest. The easy slice checks that gains on hard prompts are not bought by losing easy ones.

## seqverify_hard

Establishes the same on a training set restricted to prompts the starting policy almost never solves. Here small response groups most often carry no reward variation. This probes whether the advantage of a method grows when learning signal is scarce. Same metrics and evaluation prompts as `seqverify_main`.
