"""Batch scenario helpers for tier-4+ examples."""


def build_batch_prompts(base_prompt: str) -> list[str]:
    """Generate a small set of richer prompts from one base request."""
    return [
        base_prompt,
        f"{base_prompt} Include a stressed-scenario branch and rollback criteria.",
        f"{base_prompt} Add owner-level accountability and time windows.",
    ]
