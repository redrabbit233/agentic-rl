"""Compatibility guard for the unavailable OpenHands 0.13 E2B SDK.

OpenHands 0.13 imports its optional E2B runtime at package import time.  Its
locked E2B 0.17.1 distribution is no longer available from PyPI, while this
project deliberately uses the Docker runtime.  Install a minimal import shim
only when that legacy module path is unavailable.  Any accidental E2B use
fails closed rather than silently selecting a different sandbox.
"""
from __future__ import annotations

import sys
import types


def ensure_e2b_import_compat() -> bool:
    try:
        from e2b.sandbox.exception import TimeoutException  # noqa: F401
        return False
    except ModuleNotFoundError:
        pass

    class DisabledE2BSandbox:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("E2B runtime is disabled; use the project Rootless Docker runtime")

    class TimeoutException(Exception):
        pass

    e2b = types.ModuleType("e2b")
    sandbox = types.ModuleType("e2b.sandbox")
    exception = types.ModuleType("e2b.sandbox.exception")
    e2b.Sandbox = DisabledE2BSandbox
    sandbox.exception = exception
    exception.TimeoutException = TimeoutException
    sys.modules.update({"e2b": e2b, "e2b.sandbox": sandbox, "e2b.sandbox.exception": exception})
    return True


if __name__ == "__main__":
    substituted = ensure_e2b_import_compat()
    from openhands.agenthub.codeact_agent import CodeActAgent
    print(f"CodeActAgent import OK; e2b_import_compat={substituted}; class={CodeActAgent.__name__}")
