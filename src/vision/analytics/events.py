from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict


@dataclass(frozen=True)
class AnalyticsEvent:
    name: str
    payload: Dict[str, Any]
    timestamp: datetime

    @staticmethod
    def now(name: str, payload: Dict[str, Any] | None = None) -> "AnalyticsEvent":
        return AnalyticsEvent(
            name=name,
            payload=payload or {},
            timestamp=datetime.now(timezone.utc),
        )


class AnalyticsClient:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def send(self, event: AnalyticsEvent) -> None:
        if not self.enabled:
            return
        # Placeholder: print to stdout or extend to send to a backend
        print(f"[Analytics] {event.timestamp.isoformat()} {event.name} {event.payload}")

