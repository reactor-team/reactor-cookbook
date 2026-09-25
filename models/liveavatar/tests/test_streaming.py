"""CPU checks for the native streaming adapter and its serving profiles."""

import ast
from pathlib import Path

import pytest

from liveavatar_stage_packing import stage_groups
from liveavatar_turbo import turbo_plan

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("turbo,world_size", [("1", 3), ("0", 5)])
def test_profiles_keep_four_steps_and_dedicated_decoder(monkeypatch, turbo, world_size):
    monkeypatch.setenv("LIVEAVATAR_TURBO", turbo)
    monkeypatch.delenv("LIVEAVATAR_STEPS", raising=False)
    plan = turbo_plan()
    assert plan["world_size"] == world_size
    assert plan["sampling_steps"] == 4
    assert plan["output_rank"] == world_size - 1
    assert plan["num_gpus_dit"] == world_size - 1
    assert [stage for group in stage_groups(world_size) for stage in group] == [
        0,
        1,
        2,
        3,
    ]


def test_adapter_inherits_loading_and_cache_helpers():
    tree = ast.parse((ROOT / "liveavatar_streaming.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    assert cls.bases[0].id == "WanS2V"
    assert [n.name for n in cls.body if isinstance(n, ast.FunctionDef)] == ["generate"]
    generate = cls.body[0]
    assert "chunk_callback" in [arg.arg for arg in generate.args.args]
    assert any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "chunk_callback"
        for n in ast.walk(generate)
    )
    compile(tree, str(ROOT / "liveavatar_streaming.py"), "exec")


def test_workspace_has_no_source_rewrite_or_patch_files():
    assert not list(ROOT.rglob("*.patch"))
    for name in (
        "liveavatar_assets.py",
        "liveavatar_stage_packing.py",
        "liveavatar_streaming.py",
    ):
        tree = ast.parse((ROOT / name).read_text())
        calls = [
            n.func.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        ]
        assert "exec" not in calls
        assert "compile" not in calls
