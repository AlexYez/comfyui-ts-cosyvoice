"""
Cooperative cancellation for long-running TS CosyVoice nodes.

ComfyUI's Cancel button sets a global flag; it does not kill the worker. A node
that never reads that flag runs to completion, which for this pack meant a
voice-conversion job over a long recording — potentially hours on CPU — could not
be stopped at all. Every loop that can run for more than a moment calls
``raise_if_interrupted()`` between iterations.

Two properties matter and are easy to get wrong:

- ``InterruptProcessingException`` derives from ``BaseException``, so the nodes'
  ``except Exception`` handlers do not swallow it while their ``finally`` blocks
  still run and clean up temp files. Do not widen those handlers to bare
  ``except:``.
- ``throw_exception_if_processing_interrupted()`` clears the flag as it raises,
  so it is safe to call from nested loops: the first check to see the flag wins.
"""

from __future__ import annotations


def raise_if_interrupted() -> None:
    """
    Raise ComfyUI's interrupt exception if the user asked to cancel.

    No-ops when ``comfy`` is not importable, so helpers that use this stay
    testable outside a ComfyUI process.

    Raises:
        comfy.model_management.InterruptProcessingException: if Cancel was pressed.
    """
    try:
        import comfy.model_management as model_management
    except ImportError:
        return

    # Deliberately outside the try/except: the interrupt exception must
    # propagate, not be mistaken for a missing dependency.
    model_management.throw_exception_if_processing_interrupted()
