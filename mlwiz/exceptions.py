class TerminationRequested(Exception):
    """Raised when a graceful termination has been requested."""


class ExperimentTerminated(Exception):
    """Raised by the experiment wrapper when execution is interrupted."""
