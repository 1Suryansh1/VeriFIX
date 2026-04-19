from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from verifix.core.config import DEFAULT_CONFIG, VerifixConfig


def test_default_config_instantiates_without_error() -> None:
    assert isinstance(DEFAULT_CONFIG, VerifixConfig)


def test_env_variable_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERIFIX_MCTS_ITERATIONS", "999")
    config = VerifixConfig()
    assert config.mcts_iterations == 999


def test_yaml_round_trip(tmp_path: pytest.TempPathFactory) -> None:
    config = VerifixConfig(
        mcts_iterations=750,
        fl_algorithm="dstar",
        ranking_strategy="composite",
        supported_languages=["python"],
    )
    cfg_path = tmp_path / "verifix.yaml"

    config.to_yaml(str(cfg_path))
    loaded = VerifixConfig.from_yaml(str(cfg_path))

    assert loaded == config
    assert json.dumps(loaded.model_dump())


def test_validator_rejects_low_mcts_iterations() -> None:
    with pytest.raises(ValidationError):
        VerifixConfig(mcts_iterations=5)


def test_validator_rejects_unknown_fl_algorithm() -> None:
    with pytest.raises(ValidationError):
        VerifixConfig(fl_algorithm="unknown")


def test_validator_rejects_unknown_v3_rollout_mode() -> None:
    with pytest.raises(ValidationError):
        VerifixConfig(v3_rollout_mode="invalid")


def test_validator_rejects_non_positive_v3_weight() -> None:
    with pytest.raises(ValidationError):
        VerifixConfig(v3_beta_critic=0.0)


def test_validator_rejects_invalid_v3_rollout_depth() -> None:
    with pytest.raises(ValidationError):
        VerifixConfig(v3_min_rollout_depth=0)


def test_validator_rejects_invalid_v3_critic_threshold() -> None:
    with pytest.raises(ValidationError):
        VerifixConfig(v3_critic_threshold=1.5)


def test_validator_rejects_invalid_parallel_validation_workers() -> None:
    with pytest.raises(ValidationError):
        VerifixConfig(parallel_validation_workers=0)


def test_validator_rejects_negative_seed_values() -> None:
    with pytest.raises(ValidationError):
        VerifixConfig(random_seed=-1)

    with pytest.raises(ValidationError):
        VerifixConfig(fuzzer_seed=-10)
