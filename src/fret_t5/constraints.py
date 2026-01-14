"""DEPRECATED: This module has been merged into constrained_generation.py.

All exports are re-exported from constrained_generation for backwards compatibility.
Please update your imports to use:
    from fret_t5.constrained_generation import TabConstraintProcessor, build_v3_constraint_processor
or:
    from fret_t5 import TabConstraintProcessor, build_v3_constraint_processor
"""

import warnings

warnings.warn(
    "fret_t5.constraints is deprecated. "
    "Import from fret_t5.constrained_generation or fret_t5 instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export for backwards compatibility
from .constrained_generation import (
    TabConstraintProcessor,
    build_v3_constraint_processor,
    ConstraintFn,
)

__all__ = ["TabConstraintProcessor", "build_v3_constraint_processor", "ConstraintFn"]
