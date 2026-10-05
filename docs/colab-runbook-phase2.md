# Colab runbook: Phase 2 ablations (hand-off brief for a browser-driving Claude)

Paste everything below the line into the Claude desktop app (with Claude in Chrome or computer use enabled).

---

You're operating Google Colab in my Chrome on my behalf. Use the Chrome profile signed in as
**petegordonusa@gmail.com**. If you're not in that profile/account, stop and ask me to switch.

**Goal:** run the Phase 2 notebook of my `olmo3-embed` project on an A100 and keep it running until it finishes,
reporting progress to me. Don't edit any code. If anything differs from what's described here, stop and tell me
rather than improvising.

1. Open https://colab.research.google.com/github/petegordon/olmo3-embed/blob/main/notebooks/phase2_ablations_colab.ipynb
   (private repo; if Colab asks to authorize GitHub, ask me to approve it).
2. Runtime → Change runtime type → **A100 GPU** → Save. If A100 isn't offered or gets refused, stop and tell me.
   Don't fall back to L4 or T4.
3. Open the Secrets panel (key icon, left sidebar). Confirm a secret named `GH_TOKEN` exists with notebook access ON.
   Never reveal or copy its value. If it's missing, stop and ask me.
4. Runtime → **Run all**.
5. When the setup cell asks to **connect Google Drive**, show me the consent screen and let me approve it. Don't
   click through Google OAuth consent yourself.
6. Check the first code cell's output: the line `GPU: ...` must say A100 (or H100). If it says WARNING, interrupt
   (Runtime → Interrupt execution) and tell me.
7. The data cell builds training data once (~10 min) or prints `data cached on Drive`.
8. **Timing cell:** wait for the line like `X s/step (laptop: 46) -> ~N min per arm -> ~H h for 5 arms`. Report that
   line to me verbatim. If the projected total is over 4 h, interrupt and ask me before continuing; otherwise
   let it run.
9. The main loop trains 5 arms: p2-a0-base, p2-a1-last, p2-a2-bidir, p2-a4-neg1, p2-a6-noinstr. After each one,
   the output shows `pushed <sha> to colab-results`. Tell me each time one is pushed (arm name + time).
10. Keep the tab open and the session alive. Every ~20–30 min, check the notebook is still running (spinner on the
    loop cell, new log lines like `{'step': ..., 'loss': ...}`). If Colab shows "Are you still there?" or a
    reconnect prompt, click to stay/reconnect.
11. **If it disconnects or the runtime is recycled:** reconnect, make sure the runtime type is still A100, and
    **Run all** again. It resumes from Drive checkpoints and skips finished work. Tell me it happened.
12. **On any error** (red traceback, `train failed`, `OutOfMemoryError`, etc.): don't retry more than once. Copy the
    last ~40 lines of that cell's output and give them to me.
13. When the loop finishes, the next cell prints a summary (lines with `olmo3-embed__p2-...` and `avg`). Copy that
    whole output to me. **Do not run the Step 2 (7B) cell**: `ARMS_7B` stays empty until I decide.
14. Leave the Colab tab open at the end, and don't disconnect or delete the runtime (I'll do it after review).

Never: change notebook code, change secrets, upgrade or buy Colab plans or compute units, share the notebook, or run
anything other than what's described above.
