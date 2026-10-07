"""Synthetic-query plumbing: prompt building and robust parsing of generator replies (no LLM)."""
from olmo3_embed.data.synth import SHAPES, build_prompt, parse


def test_prompt_contains_ids_and_clips_passages():
    p = build_prompt("entity", [{"id": "entity-1", "passage": "x" * 5000}])
    assert "[id: entity-1]" in p and "keyword-style" in p and len(p) < 3000


def test_parse_ignores_echoed_template_and_bad_items():
    echoed = build_prompt("argument", [{"id": "a-1", "passage": "Some speech."}])  # contains {"items": [... "<text>"]}
    reply = echoed + '\ncodex\n{"items": [{"id": "a-1", "query": "Should zoos be banned?"}, {"id": "zz", "query": "x y z"}]}'
    assert parse(reply, ["a-1"]) == {"a-1": "Should zoos be banned?"}
    assert parse('{"items": [{"id": "a-1", "query": "<text>"}]}', ["a-1"]) == {}  # template placeholder rejected
    assert parse("no json", ["a-1"]) == {}


def test_every_shape_has_a_task_and_source():
    for shape, (task, src, instr) in SHAPES.items():
        assert task.startswith("Synth") and "/" in src and len(instr) > 50
