"""Sandbox profile and registry policy builders."""

from pathlib import Path

from afk.tools import (
    SandboxProfile,
    build_registry_output_limit_middleware,
    build_registry_sandbox_policy,
)


def build_profile(root_dir: Path) -> SandboxProfile:
    return SandboxProfile(
        profile_id="runtime-fileops",
        allowed_paths=[str(root_dir)],
        denied_paths=["/etc"],
        max_output_chars=180,
    )


def build_policy_and_middleware(root_dir: Path):
    profile = build_profile(root_dir)
    policy = build_registry_sandbox_policy(profile=profile, cwd=Path.cwd())
    middleware = build_registry_output_limit_middleware(profile=profile)
    return profile, policy, middleware
