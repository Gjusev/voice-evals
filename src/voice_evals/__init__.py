"""voice-evals: evaluation harness for voice agents.

Replay-mode scoring of recorded calls: live WER, latency budgets,
barge-in behavior and judge-based task outcomes, with CI gates.
"""

from .evaluate import evaluate
from .types import (
    CallRecord,
    Interruption,
    Scenario,
    StageTimings,
    VoiceEvalResult,
    load_dataset,
)

__version__ = "0.2.1"

__all__ = [
    "CallRecord",
    "Interruption",
    "Scenario",
    "StageTimings",
    "VoiceEvalResult",
    "__version__",
    "evaluate",
    "load_dataset",
]
