"""Opt-in bootstrap. Put this directory on PYTHONPATH for spawn-style workers."""
import os
import sys

if os.environ.get("HANG_MSPTI_DIR"):
    # This bootstrap runs before app imports, so libmspti.so must already be
    # in LD_PRELOAD. Forked children require an explicit post-fork start hook.
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from hang_mspti import start

    start(os.environ["HANG_MSPTI_DIR"], os.environ["HANG_MSPTI_LIBRARY"],
          interval_s=float(os.environ.get("HANG_MSPTI_INTERVAL_S", "0.5")),
          capacity=int(os.environ.get("HANG_MSPTI_CAPACITY", "262144")),
          control_file=os.environ.get("HANG_MSPTI_ENABLE_FILE", "/home/enable_prof"))
