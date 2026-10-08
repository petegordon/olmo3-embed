from olmo3_embed.train import Config, EXTRA_INSTRUCTIONS, dropped_instruction, query_text, task_instruction


def test_dropout_rate_and_stability():
    qs = [f"query number {i}" for i in range(20_000)]
    rate = sum(dropped_instruction(q, 0.2) for q in qs) / len(qs)
    assert 0.18 < rate < 0.22
    assert [dropped_instruction(q, 0.2) for q in qs[:100]] == [dropped_instruction(q, 0.2) for q in qs[:100]]
    assert not any(dropped_instruction(q, 0.0) for q in qs[:1000])


def test_query_text_per_task():
    cfg = Config(instructions=True, instruction_mode="per_task", instruction_dropout=0.0)
    t = query_text({"task": "SynthCounterArgument", "query": "Zoos should be banned."}, cfg)
    assert t == f"Instruct: {EXTRA_INSTRUCTIONS['SynthCounterArgument']}\nQuery: Zoos should be banned."
    assert task_instruction("SynthArgument") == EXTRA_INSTRUCTIONS["SynthArgument"]
    off = Config(instructions=False)
    assert query_text({"task": "SynthArgument", "query": "q"}, off) == "q"
