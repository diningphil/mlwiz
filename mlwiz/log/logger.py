import os
from pathlib import Path


class Logger:
    r"""
    Provide minimal file-based logging utilities.

    Args:
        filepath (str): Path to the file to write.
        mode (str): File open mode: ``'w'`` (truncate on each write) or
            ``'a'`` (append).
        debug (bool): Debug flag stored on the instance. This class does not
            currently branch on it, but callers may use it to decide whether
            to emit logs.

    """

    def __init__(self, filepath, mode, debug):
        r"""
        Initialize the logger and ensure the log directory exists.

        Args:
            filepath (str | pathlib.Path): Log file path.
            mode (str): File open mode: ``'w'`` or ``'a'``.
            debug (bool): Debug flag stored on the instance.

        Raises:
            ValueError: If ``mode`` is not ``'w'`` or ``'a'``.

        Side effects:
            Creates the parent directory of ``filepath`` if it does not exist.
        """
        self.debug = debug
        self.filepath = Path(filepath)
        if not os.path.exists(self.filepath.parent):
            os.makedirs(self.filepath.parent)

        if mode not in ["w", "a"]:
            raise ValueError("Mode must be one of w or a")
        else:
            self.mode = mode

    def log(self, content):
        r"""
        Write a single line to the configured log file.

        Args:
            content (str): Line content to write. A trailing newline is added.

        Side effects:
            Writes to disk using ``self.mode``. Note that ``mode='w'`` will
            truncate the file on every call to this method.
        """
        try:
            with open(self.filepath, self.mode) as f:
                f.write(content + "\n")
        except Exception as e:
            print(e)
