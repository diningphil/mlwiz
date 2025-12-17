"""Custom exceptions used by MLWiz.

Defines termination-related exceptions raised by the evaluator and experiment wrapper.
"""

class TerminationRequested(Exception):
    """Raised when a graceful termination has been requested."""


class ExperimentTerminated(Exception):
    """Raised by the experiment wrapper when execution is interrupted."""
