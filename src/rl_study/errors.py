"""Domain-specific errors with stable CLI meanings."""


class RLStudyError(Exception):
    """Base class for expected, user-actionable failures."""


class ConfigError(RLStudyError):
    """The supplied configuration violates the strict schema."""


class PreflightError(RLStudyError):
    """A device, dependency, model, or dataset failed preflight."""


class DownloadApprovalRequired(PreflightError):
    """A large download was blocked before network access."""


class CheckpointError(RLStudyError):
    """Checkpoint integrity or resume compatibility failed."""


class NumericError(RLStudyError):
    """Training or evaluation produced an invalid numeric value."""


class AgenticError(RLStudyError):
    """An Agentic RL trajectory violates an environment or policy contract."""


class StaleTrajectoryError(AgenticError):
    """A rollout was produced by a policy version outside the allowed lag."""


class RetokenizationDriftError(AgenticError):
    """Decoded action text no longer maps to the original rollout token IDs."""
