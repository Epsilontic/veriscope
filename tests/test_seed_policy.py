# tests/test_seed_policy.py
from __future__ import annotations

from typing import Mapping, Sequence, Set

import pytest

from veriscope.runners.seed_policy import seeds_for_eval_from_env

ENV_RELEVANT = ("SCAR_SMOKE", "SCAR_EVAL_SPLIT")


@pytest.fixture
def mock_config() -> Mapping[str, Sequence[int]]:
    return {"seeds_eval": [511, 512], "seeds_calib": [512, 513]}


@pytest.mark.parametrize(
    "env_vars, expected",
    [
        ({"SCAR_SMOKE": "1"}, {511, 512}),  # smoke default -> eval
        ({"SCAR_SMOKE": "1", "SCAR_EVAL_SPLIT": "both"}, {511, 512, 513}),
        ({}, {511, 512}),  # default is eval when not set (consistent with your function)
        ({"SCAR_EVAL_SPLIT": "calib"}, {512, 513}),
        ({"SCAR_EVAL_SPLIT": " BOTH "}, {511, 512, 513}),
    ],
)
def test_seeds_for_eval_from_env(monkeypatch, mock_config, env_vars, expected: Set[int]):
    for k in ENV_RELEVANT:
        monkeypatch.delenv(k, raising=False)
    for k, v in env_vars.items():
        monkeypatch.setenv(k, v)

    out = seeds_for_eval_from_env(mock_config)

    assert len(out) == len(set(out)), "no duplicates"
    assert set(out) == expected
