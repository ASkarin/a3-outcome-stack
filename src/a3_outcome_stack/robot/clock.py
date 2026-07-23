"""Injectable monotonic clocks used for ordering and watchdog decisions."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Protocol

from a3_outcome_stack.ops.errors import ValidationError


class Clock(Protocol):
    @property
    def domain_id(self) -> str: ...

    def now_ns(self) -> int: ...


class MonotonicClock:
    def __init__(self, domain_id: str | None = None):
        self._domain_id = domain_id or f"host-monotonic-{uuid.uuid4()}"

    @property
    def domain_id(self) -> str:
        return self._domain_id

    def now_ns(self) -> int:
        return time.monotonic_ns()


@dataclass
class ManualClock:
    """Deterministic test clock. It never regresses through its public API."""

    current_ns: int = 0
    domain_id: str = "manual-clock-v1"

    def __post_init__(self) -> None:
        if self.current_ns < 0 or not self.domain_id:
            raise ValidationError(
                "manual clock requires a non-negative time and domain_id"
            )

    def now_ns(self) -> int:
        return self.current_ns

    def advance_ns(self, amount_ns: int) -> int:
        if not isinstance(amount_ns, int) or amount_ns < 0:
            raise ValidationError("clock advance must be a non-negative integer")
        self.current_ns += amount_ns
        return self.current_ns

    def set_ns(self, value_ns: int) -> None:
        if not isinstance(value_ns, int) or value_ns < self.current_ns:
            raise ValidationError("manual clock cannot move backwards")
        self.current_ns = value_ns
