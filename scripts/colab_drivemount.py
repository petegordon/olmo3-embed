"""`colab drivemount` for agents: same flow, but no terminal needed and a longer wait for the consent.

The Colab CLI's drivemount waits for Enter on /dev/tty after printing the Drive consent URL, and Colab's
drive.mount() gives up after 120 s. An agent relaying the URL to Pete has neither a tty nor 2 minutes. This wrapper
runs the CLI's own drivemount with two patches:
  - "Press Enter" waits for a signal file instead (touch it once the consent is approved);
  - drive.mount() and the CLI wait 20 min instead of 2 and 10.
The consent is per runtime: every new VM needs a fresh approval.

Run it with the CLI's interpreter:
  ~/.local/share/uv/tools/google-colab-cli/bin/python scripts/colab_drivemount.py SESSION SIGNAL_FILE
"""
import builtins
import os
import sys
import time

from colab_cli import cli
from colab_cli.commands import automation

session, signal_file = sys.argv[1], sys.argv[2]


class _WaitForSignal:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def readline(self):
        print(f'\n[colab_drivemount] waiting for {signal_file} (20 min)', flush=True)
        deadline = time.time() + 1200
        while not os.path.exists(signal_file):
            if time.time() > deadline:
                raise TimeoutError('no Drive consent within 20 min')
            time.sleep(2)
        os.remove(signal_file)
        return '\n'


_open = builtins.open
builtins.open = lambda f, *a, **k: _WaitForSignal() if f == '/dev/tty' else _open(f, *a, **k)

automation.INTERACTIVE_AUTOMATION_TIMEOUT_SEC = 1260  # the CLI's own wait, a bit longer than ours
_run = automation.run_automation
automation.run_automation = lambda name, op, code, **k: _run(
    name, op, code.replace("drive.mount('/content/drive')", "drive.mount('/content/drive', timeout_ms=1200000)"), **k)

sys.argv = ['colab', 'drivemount', '-s', session]
cli.main()
