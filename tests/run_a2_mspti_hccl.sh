#!/usr/bin/env bash
set -euo pipefail
cd /tmp/HangAnalyzer-validation
export LD_PRELOAD=/usr/local/Ascend/cann-9.1.0/lib64/libmspti.so
export PYTHONPATH=.
export HANG_MSPTI_LIBRARY=/tmp/hang-mspti-build/libhangmspti.so
export HANG_TEST_OUTPUT=/tmp/hang-mspti-hccl-$$
export MASTER_PORT=29587
LOCAL_RANK=0 python tests/a2_mspti_hccl_smoke.py > /tmp/hang-mspti-rank0-$$.log 2>&1 &
p0=$!
LOCAL_RANK=1 python tests/a2_mspti_hccl_smoke.py > /tmp/hang-mspti-rank1-$$.log 2>&1 &
p1=$!
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
cat /tmp/hang-mspti-rank0-$$.log /tmp/hang-mspti-rank1-$$.log
exit "$rc"
