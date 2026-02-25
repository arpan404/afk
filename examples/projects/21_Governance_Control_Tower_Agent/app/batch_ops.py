"""Batch prompt helpers for governance-scale workflows."""


def build_batch_prompts(base_prompt: str) -> list[str]:
    """Generate increasingly strict governance-oriented prompts."""
    return [
        base_prompt,
        f"{base_prompt} Add scenario stress outcomes and mitigation sequencing.",
        f"{base_prompt} Include compliance controls and audit evidence requirements.",
        f"{base_prompt} Add executive summary with measurable KPIs and risk appetite fit.",
    ]
