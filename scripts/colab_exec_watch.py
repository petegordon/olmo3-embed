"""`colab exec` for long cells: notices a dropped connection within ~30 s, whatever --timeout is.

The CLI's client sleeps on one Event until a message arrives or the whole --timeout runs out, and only checks the
connection after it wakes. A long --timeout (needed: a cell longer than it fails with "Timeout waiting for reply")
therefore hides a dropped websocket for hours, and Colab reclaims a runtime with no client after ~20 min.
This wrapper caps each wait at 30 s, so the loop re-checks the connection and raises "Connection was lost." quickly;
the deadline is still computed from --timeout. A driver can then re-exec (training resumes from its checkpoint).

Run it with the CLI's interpreter, with `colab exec`'s arguments:
  ~/.local/share/uv/tools/google-colab-cli/bin/python scripts/colab_exec_watch.py -s SESSION -f NB.ipynb --timeout 36000
"""
import sys
import threading

from jupyter_kernel_client import wsclient
from colab_cli import cli

POLL_SEC = 30.0


class _PollingEvent(threading.Event):
    def wait(self, timeout=None):
        return super().wait(POLL_SEC if timeout is None else min(timeout, POLL_SEC))


wsclient.Event = _PollingEvent

sys.argv = ['colab', 'exec', *sys.argv[1:]]
cli.main()
