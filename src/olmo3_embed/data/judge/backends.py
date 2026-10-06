"""Judge backends. Each takes a list of prompts and returns the raw response texts.

  vllm    Olmo-3.1-32B-Instruct (or any HF chat model) offline on a Colab GPU; batched, greedy
  claude  Claude Code CLI on Pete's Max plan:  claude -p --output-format json --model <model>
  codex   OpenAI Codex CLI on ChatGPT Plus:    codex exec <prompt>
  gemini  Antigravity CLI on Google AI Pro:    agy -p <prompt>   (hangs without a TTY on Windows: run from WSL/Linux)
"""
import json
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


class CLI:
    """Base for subscription CLIs: one subprocess per prompt (prompt on stdin), sequential to respect usage windows."""
    name, exe = "cli", ""

    def __init__(self, model: str | None = None, timeout: int = 600):
        if not shutil.which(self.exe):
            raise RuntimeError(f"{self.exe!r} not on PATH: install and sign in to it first")
        self.model, self.timeout = model, timeout

    def cmd(self) -> list[str]:
        raise NotImplementedError

    def text(self, stdout: str) -> str:
        return stdout

    def __call__(self, prompts: list[str]) -> list[str]:
        out = []
        for p in prompts:
            r = subprocess.run(self.cmd(), input=p, capture_output=True, text=True, encoding="utf-8",
                               timeout=self.timeout)
            if r.returncode != 0:
                raise RuntimeError(f"{self.exe} exited {r.returncode}: {r.stderr[-500:]}")
            out.append(self.text(r.stdout))
        return out


class Claude(CLI):
    name, exe = "claude", "claude"

    def cmd(self):
        return ["claude", "-p", "--output-format", "json"] + (["--model", self.model] if self.model else [])

    def text(self, stdout):
        return json.loads(stdout).get("result", "")


class Codex(CLI):
    name, exe = "codex", "codex"

    def cmd(self):
        return ["codex", "exec", "-"] + (["--model", self.model] if self.model else [])


class Gemini(CLI):
    name, exe = "gemini", "agy"

    def cmd(self):
        return ["agy", "-p"] + (["--model", self.model] if self.model else [])


BACKENDS = {"vllm": VLLM, "claude": Claude, "codex": Codex, "gemini": Gemini}
