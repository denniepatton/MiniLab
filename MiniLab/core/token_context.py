"""Token context metadata.

We attach lightweight metadata (workflow, trigger) to TokenAccount debits
using contextvars. This enables accurate, non-hardcoded accounting and
budget calibration based on empirical measurements.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional


_current_workflow: ContextVar[Optional[str]] = ContextVar("minilab_current_workflow", default=None)
_current_trigger: ContextVar[Optional[str]] = ContextVar("minilab_current_trigger", default=None)


def get_workflow() -> Optional[str]:
    return _current_workflow.get()


def get_trigger() -> Optional[str]:
    return _current_trigger.get()


@contextmanager
def token_context(*, workflow: Optional[str] = None, trigger: Optional[str] = None) -> Iterator[None]:
    tokens = []
    if workflow is not None:
        tokens.append((_current_workflow, _current_workflow.set(workflow)))
    if trigger is not None:
        tokens.append((_current_trigger, _current_trigger.set(trigger)))
    try:
        yield
    finally:
        for var, tok in reversed(tokens):
            try:
                var.reset(tok)
            except Exception:
                pass
