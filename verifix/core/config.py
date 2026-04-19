from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class VerifixConfig(BaseSettings):
    # MCTS Parameters
    mcts_iterations: int = 500
    mcts_max_depth: int = 5
    mcts_exploration_constant: float = 1.414
    mcts_rollout_policy: str = "random"
    mcts_time_budget_seconds: float = 120.0

    # Search Parameters
    max_candidates_per_node: int = 10
    beam_fallback_k: int = 3
    max_patch_candidates: int = 50
    random_seed: int = 20260404

    # Validation Parameters
    test_timeout_seconds: float = 30.0
    compile_timeout_seconds: float = 15.0
    max_validations: int = 200
    validate_all_nodes: bool = False
    sandbox_execution: bool = True
    parallel_validation_workers: int = 4

    # Fault Localization
    fl_algorithm: str = "ochiai"
    fl_top_n_lines: int = 10
    fl_use_existing_coverage: bool = False

    # Patch Ranking
    ranking_strategy: str = "validation_score"
    dedup_identical_patches: bool = True

    # Paths
    working_dir: str = "/tmp/verifix_workspace"
    benchmark_data_dir: str = "./data/benchmarks"
    results_output_dir: str = "./results"

    # Language Support
    supported_languages: list[str] = Field(default_factory=lambda: ["python", "java"])
    python_executable: str = sys.executable
    java_executable: str = "java"
    javac_executable: str = "javac"

    # Logging
    log_level: str = "INFO"
    log_search_tree: bool = False
    verbose: bool = False

    # V3 Experimental Controls
    v3_enabled: bool = False
    v3_rollout_mode: str = "concrete"
    v3_alpha_jepa: float = 1.0
    v3_beta_critic: float = 10.0
    v3_beta_localization: float = 1.0
    v3_beta_policy: float = 1.0
    v3_split_seed: int = 20260404
    v3_min_rollout_depth: int = 3
    v3_branch_per_state: int = 3
    v3_critic_threshold: float = 0.45
    v3_candidate_node_weight: float = 0.2
    v3_candidate_action_weight: float = 0.8
    fuzzer_seed: int = 42

    model_config = SettingsConfigDict(
        env_prefix="VERIFIX_",
        env_file=".env",
        case_sensitive=False,
    )

    @field_validator("mcts_iterations")
    @classmethod
    def validate_mcts_iterations(cls, value: int) -> int:
        if not 10 <= value <= 100000:
            raise ValueError("mcts_iterations must be in [10, 100000]")
        return value

    @field_validator("mcts_max_depth")
    @classmethod
    def validate_mcts_max_depth(cls, value: int) -> int:
        if not 1 <= value <= 20:
            raise ValueError("mcts_max_depth must be in [1, 20]")
        return value

    @field_validator("fl_top_n_lines")
    @classmethod
    def validate_fl_top_n_lines(cls, value: int) -> int:
        if value < 1:
            raise ValueError("fl_top_n_lines must be >= 1")
        return value

    @field_validator("mcts_exploration_constant")
    @classmethod
    def validate_mcts_exploration_constant(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("mcts_exploration_constant must be > 0.0")
        return value

    @field_validator("fl_algorithm")
    @classmethod
    def validate_fl_algorithm(cls, value: str) -> str:
        allowed = {"ochiai", "tarantula", "dstar"}
        if value not in allowed:
            raise ValueError("fl_algorithm must be one of ['ochiai', 'tarantula', 'dstar']")
        return value

    @field_validator("ranking_strategy")
    @classmethod
    def validate_ranking_strategy(cls, value: str) -> str:
        allowed = {"validation_score", "edit_count", "composite"}
        if value not in allowed:
            raise ValueError(
                "ranking_strategy must be one of ['validation_score', 'edit_count', 'composite']"
            )
        return value

    @field_validator("v3_rollout_mode")
    @classmethod
    def validate_v3_rollout_mode(cls, value: str) -> str:
        allowed = {"concrete", "latent", "hybrid"}
        if value not in allowed:
            raise ValueError("v3_rollout_mode must be one of ['concrete', 'latent', 'hybrid']")
        return value

    @field_validator("v3_alpha_jepa", "v3_beta_critic", "v3_beta_localization", "v3_beta_policy")
    @classmethod
    def validate_v3_loss_weight(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("V3 loss weights must be > 0.0")
        return value

    @field_validator("v3_min_rollout_depth", "v3_branch_per_state")
    @classmethod
    def validate_v3_positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("V3 rollout depth and branch settings must be >= 1")
        return value

    @field_validator("v3_critic_threshold")
    @classmethod
    def validate_v3_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("v3_critic_threshold must be in [0.0, 1.0]")
        return value

    @field_validator("v3_candidate_node_weight", "v3_candidate_action_weight")
    @classmethod
    def validate_v3_candidate_weight(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("V3 candidate score weights must be in [0.0, 1.0]")
        return value

    @field_validator("parallel_validation_workers")
    @classmethod
    def validate_parallel_workers(cls, value: int) -> int:
        if not 1 <= value <= 32:
            raise ValueError("parallel_validation_workers must be in [1, 32]")
        return value

    @field_validator("random_seed", "fuzzer_seed")
    @classmethod
    def validate_non_negative_seed(cls, value: int) -> int:
        if value < 0:
            raise ValueError("seed values must be >= 0")
        return value

    @classmethod
    def from_yaml(cls, path: str) -> VerifixConfig:
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("YAML configuration must contain a mapping at the root")
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        payload = self.model_dump()
        with Path(path).open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=True)

    def __repr__(self) -> str:
        current = self.model_dump()
        non_defaults: dict[str, Any] = {}

        for field_name, field_info in self.__class__.model_fields.items():
            if field_info.default_factory is not None:
                default_value = field_info.default_factory()
            else:
                default_value = field_info.default

            if current[field_name] != default_value:
                non_defaults[field_name] = current[field_name]

        args = ", ".join(f"{key}={value!r}" for key, value in sorted(non_defaults.items()))
        return f"{self.__class__.__name__}({args})"


DEFAULT_CONFIG = VerifixConfig()


class QuixBugsConfig(VerifixConfig):
    """Canonical config for QuixBugs comparisons across V1/V2/V3 modes."""

    mcts_iterations: int = 200
    mcts_max_depth: int = 2
    mcts_exploration_constant: float = 2.0
    mcts_time_budget_seconds: float = 60.0
    max_candidates_per_node: int = 15
    max_validations: int = 100
    validate_all_nodes: bool = True
    fl_top_n_lines: int = 5
    python_executable: str = sys.executable


__all__ = ["VerifixConfig", "DEFAULT_CONFIG", "QuixBugsConfig"]
