"""Shared reasoning evaluation package.

Canonical location:
    eval/judge_reasoning/
"""

__all__ = ["LLMJudgeEvaluator", "config"]


def __getattr__(name):
    # Lazy imports keep `import eval.judge_reasoning` lightweight.
    if name == "LLMJudgeEvaluator":
        from .llm_judge_eval import LLMJudgeEvaluator

        return LLMJudgeEvaluator
    if name == "config":
        from . import config

        return config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
