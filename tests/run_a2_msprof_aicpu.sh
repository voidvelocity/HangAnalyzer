#!/usr/bin/env bash
set -euo pipefail
cd /tmp/HangAnalyzer-validation
export PYTHONPATH=.
export MASTER_PORT=29589
out=/tmp/hang-msprof-aicpu-$$
mkdir -p "$out"
LOCAL_RANK=0 msprof --output="$out/rank0" --task-time=on --aicpu=on --hccl=on \
  python tests/a2_hccl_baseline.py >"$out/rank0.log" 2>&1 &
p0=$!
LOCAL_RANK=1 msprof --output="$out/rank1" --task-time=on --aicpu=on --hccl=on \
  python tests/a2_hccl_baseline.py >"$out/rank1.log" 2>&1 &
p1=$!
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
echo "output=$out"
tail -15 "$out/rank0.log"
tail -15 "$out/rank1.log"
exit "$rc"
