#!/bin/bash
# Download all data and models needed for this environment.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: 0 GB
#
# This environment has no external artifacts: the problem instance (base-policy
# weights, training prompts with verifier hashes, visible dev prompts) is
# agent-authored, small, and committed to git under data/seqverify/ and
# eval/train/. The only runtime dependency is numpy, installed by
# environment/setup.sh. The script is kept idempotent so it is safe to call.
set -e
cd /home/user
mkdir -p data/downloads
for f in data/seqverify/problem.json data/seqverify/train_main.json data/seqverify/train_hard.json eval/train/dev_set.json; do
  if [ ! -s "$f" ]; then
    echo "missing committed problem file: $f (restore it from git: git checkout -- $f)" >&2
    exit 1
  fi
done
echo "All problem files present; nothing to download."
