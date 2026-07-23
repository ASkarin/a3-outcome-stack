"""Compatibility namespace for Stage 0/1A A3 OutcomeStack imports.

New code must import :mod:`a3_outcome_stack`. This alias remains intentionally
small so historical commands can still resolve the canonical package.
"""

from a3_outcome_stack import __version__
from a3_outcome_stack import __path__ as _canonical_path

__path__ = _canonical_path

__all__ = ["__version__"]
