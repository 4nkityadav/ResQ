"""
resq_ai.core.event_bus
======================
Central asynchronous-style, deterministic synchronous event bus for ResQ-AI.
Enforces the single data flow pipeline:
  Sensor Telemetry
    -> HMM Hazard Tracker
      -> Bayesian Network Triage (Ambient Hazard + Caller Evidence)
        -> D* Lite Dynamic Edge & Cost Routing
          -> Fleet Dispatcher (Simulated Annealing / ALNS)
            -> Field Execution & Vehicle Movement
              -> WorldState Update
"""

from __future__ import annotations
from typing import Dict, List, Callable, Any, Optional
import time

from resq_ai.core.models import EventType, SimEvent
from resq_ai.core.logger import ResQLogger


EventHandler = Callable[[SimEvent], Any]


class EventBus:
    """
    Publish-subscribe event bus facilitating decoupled multi-module integration.
    Maintains a deterministic event log and supports transactional pipeline tracing.
    """

    def __init__(self):
        self._subscribers: Dict[EventType, List[EventHandler]] = {et: [] for et in EventType}
        self._history: List[SimEvent] = []
        self.logger = ResQLogger.get()

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Registers a callback handler for a specific event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def publish(self, event: SimEvent) -> List[Any]:
        """
        Dispatches an event synchronously to all registered handlers.
        Returns a list of result objects returned by the handlers.
        """
        self._history.append(event)
        handlers = self._subscribers.get(event.event_type, [])
        results = []

        for handler in handlers:
            try:
                res = handler(event)
                if res is not None:
                    results.append(res)
            except Exception as e:
                self.logger.error("EVENT_BUS", f"Error in handler {handler.__name__} for {event.event_type.value}: {e}", error=str(e))
                raise

        return results

    def get_history(self) -> List[SimEvent]:
        return list(self._history)

    def can_undo(self) -> bool:
        """Returns True if there are reversible events in history."""
        return len(self._history) > 0

    def undo(self) -> Optional[SimEvent]:
        """Pops and returns the most recent event from history."""
        if self._history:
            return self._history.pop()
        return None

    def clear(self) -> None:
        self._history.clear()
