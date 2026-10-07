"""Judge backends. Each takes a list of prompts and returns the raw response texts.

  vllm    Olmo-3.1-32B-Instruct (or any HF chat model) offline on a Colab GPU; batched, greedy
  claude  Claude Code CLI on Pete's Max plan:  claude -p --output-format json --model <model>
  codex   OpenAI Codex CLI on ChatGPT Plus:    codex exec <prompt>
  gemini  Antigravity CLI on Google AI Pro:    agy -p <prompt>   (hangs without a TTY on Windows: run from WSL/Linux)
"""
import json
import os
import shutil
import subprocess


class VLLM:
    name = "vllm"

    def __init__(self, model: str = "allenai/Olmo-3.1-32B-Instruct", max_model_len: int = 12288,
                 gpu_memory_utilization: float = 0.95, **kw):
        # 32B bf16 ≈ 64 GB of an 80 GB GPU: the rest is KV cache. A prompt is ≤ 25 candidates x 1200 chars (~8k
        # tokens) + ≤ 1200 output tokens, so 12k context fits every prompt and leaves room for concurrency.
        from vllm import LLM, SamplingParams
        self.model = model
        self.llm = LLM(model=model, dtype="bfloat16", max_model_len=max_model_len, enable_prefix_caching=True,
                       gpu_memory_utilization=gpu_memory_utilization, **kw)
        self.params = SamplingParams(temperature=0.0, max_tokens=1200)

    def __call__(self, prompts: list[str]) -> list[str]:
        msgs = [[{"role": "user", "content": p}] for p in prompts]
        return [o.outputs[0].text for o in self.llm.chat(msgs, self.params, use_tqdm=True)]


def find_exe(exe: str) -> str | None:
    """PATH first, then the per-user install dirs (npm global, Antigravity) so a fresh install works before the
    shell/IDE is restarted."""
    hit = shutil.which(exe)
    if hit:
        return hit
    for d in (os.path.join(os.environ.get("APPDATA", ""), "npm"),
              os.path.join(os.environ.get("LOCALAPPDATA", ""), "agy", "bin"), os.path.expanduser("~/.local/bin")):
        hit = shutil.which(exe, path=d)
        if hit:
            return hit
    return None


class CLI:
    """Base for subscription CLIs: one subprocess per prompt, sequential to respect usage windows.
    The prompt goes on stdin unless the subclass sets prompt_arg (then it's the last argv item)."""
    name, exe, prompt_arg = "cli", "", False

    def __init__(self, model: str | None = None, timeout: int = 900):
        self.path = find_exe(self.exe)
        if not self.path:
            raise RuntimeError(f"{self.exe!r} not found: install and sign in to it first")
        self.model, self.timeout = model, timeout

    def cmd(self) -> list[str]:
        raise NotImplementedError

    def text(self, stdout: str) -> str:
        return stdout

    max_consecutive_failures = 5  # then stop: likely auth, quota or an outage, not a one-off

    def __call__(self, prompts: list[str]) -> list[str]:
        """A failed or timed-out call returns "" (its ids stay unjudged/ungenerated and are retried on the next run),
        so one hung CLI call can't kill a multi-hour job. Several failures in a row still raise."""
        out = []
        for p in prompts:
            argv = [self.path] + self.cmd() + ([p] if self.prompt_arg else [])
            try:
                r = subprocess.run(argv, input=None if self.prompt_arg else p,
                                   stdin=subprocess.DEVNULL if self.prompt_arg else None, capture_output=True, text=True,
                                   encoding="utf-8", timeout=self.timeout)
                if r.returncode != 0:
                    raise RuntimeError(f"exited {r.returncode}: {r.stderr[-300:]}")
                out.append(self.text(r.stdout))
                self._fails = 0
            except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError) as e:
                self._fails = getattr(self, "_fails", 0) + 1
                print(f"  [{self.exe}] call failed ({self._fails} in a row): {str(e)[:200]}", flush=True)
                if self._fails >= self.max_consecutive_failures:
                    raise RuntimeError(f"{self.exe}: {self._fails} consecutive failures, stopping") from e
                out.append("")
        return out


class Claude(CLI):
    name, exe = "claude", "claude"

    def cmd(self):
        return ["-p", "--output-format", "json"] + (["--model", self.model] if self.model else [])

    def text(self, stdout):
        return json.loads(stdout).get("result", "")


class Codex(CLI):
    """codex exec: the final message on stdout, the transcript on stderr."""
    name, exe = "codex", "codex"

    def cmd(self):
        return ["exec", "-"] + (["--model", self.model] if self.model else [])


class Gemini(CLI):
    """agy -p needs the prompt as an argument (no stdin), so keep prompts well under Windows' 32k-char command line:
    judge with --batch 10. Non-TTY print mode works in agy 1.3.0 (the 1.0.6 hang, issue #318, is gone)."""
    name, exe, prompt_arg = "gemini", "agy", True

    def __init__(self, model: str | None = None, timeout: int = 420):  # agy's own --print-timeout is 5 min
        super().__init__(model, timeout)
        self.model = model or "agy-default"

    def cmd(self):
        return ["--output-format", "json", "--disable-slash-commands", "--print-timeout", "5m"] + (
            ["--model", self.model] if self.model != "agy-default" else []) + ["-p"]

    def text(self, stdout):
        return json.loads(stdout).get("response", "")


BACKENDS = {"vllm": VLLM, "claude": Claude, "codex": Codex, "gemini": Gemini}
