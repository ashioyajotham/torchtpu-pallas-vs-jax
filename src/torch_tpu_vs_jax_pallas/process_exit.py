"""Optional diagnostic process-exit workaround for PJRT teardown failures."""

from __future__ import annotations

import os
import sys


def finalize_tpu_process() -> None:
    """Normally exit cleanly; opt into the diagnostic workaround explicitly.

    Some runtime versions have faulted while destroying a loaded executable
    after all requested work and artifact writes have completed. Only setting
    TORCH_TPU_SAFE_PROCESS_EXIT=1 skips Python teardown; this is not a
    general-purpose runtime fix.
    """
    if os.environ.get("TORCH_TPU_SAFE_PROCESS_EXIT", "0") != "1":
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
