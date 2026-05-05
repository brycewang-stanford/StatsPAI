"""Regression tests for the CausalAgentBench mock runner."""

from __future__ import annotations

import json
import os
import subprocess
import sys


def _run_mock_with_hash_seed(seed: str) -> dict:
    code = r"""
import json
import sys
from pathlib import Path

root = Path.cwd()
runners = root / "tests" / "agent_bench" / "runners"
sys.path.insert(0, str(runners))

from mock_llm import run_trial, to_dict

prompt = json.loads((root / "tests" / "agent_bench" / "prompts" / "prompts.json").read_text())[0]
gold = json.loads((root / "tests" / "agent_bench" / "golds" / "golds.json").read_text())[0]
print(json.dumps(to_dict(run_trial("C1", prompt, gold, 0)), sort_keys=True))
"""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = seed
    out = subprocess.check_output([sys.executable, "-c", code], text=True, env=env)
    return json.loads(out)


def test_mock_llm_is_stable_across_python_hash_seeds():
    """The mock harness is a fixture, so Python hash randomization must not affect it."""
    assert _run_mock_with_hash_seed("1") == _run_mock_with_hash_seed("2")
