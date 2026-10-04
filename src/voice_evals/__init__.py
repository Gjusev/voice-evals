"""voice-evals: evaluation harness for voice agents.

Replay-mode scoring of recorded calls: live WER, latency budgets,
barge-in behavior and judge-based task outcomes, with CI gates.
"""

from .evaluate import evaluate
from .types import CallRecord, Interruption, Scenario, StageTimings, VoiceEvalResult

__version__ = "0.1.0"

__all__ = [
    "CallRecord",
    "Interruption",
    "Scenario",
    "StageTimings",
    "VoiceEvalResult",
    "evaluate",
    "__version__",
]
