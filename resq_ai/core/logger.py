"""
resq_ai.core.logger
===================
Structured logging for ResQ-AI with in-memory audit trail for UI/CLI inspection.
"""

from __future__ import annotations
import logging
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime


class AuditEntry:
    def __init__(self, step: int, module: str, level: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        self.timestamp = datetime.now()
        self.step = step
        self.module = module
        self.level = level
        self.message = message
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.strftime("%H:%M:%S.%f")[:-3],
            "step": self.step,
            "module": self.module,
            "level": self.level,
            "message": self.message,
            "metadata": self.metadata,
        }


class ResQLogger:
    """Singleton-style structured logger keeping recent logs in memory for the UI console."""
    
    _instance: Optional[ResQLogger] = None

    def __init__(self, max_audit_entries: int = 500):
        self.audit_log: List[AuditEntry] = []
        self.max_entries = max_audit_entries
        self.current_step = 0
        
        # Setup standard Python logger
        self._logger = logging.getLogger("ResQ-AI")
        self._logger.setLevel(logging.INFO)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S")
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

    @classmethod
    def get(cls) -> ResQLogger:
        if cls._instance is None:
            cls._instance = ResQLogger()
        return cls._instance

    def set_step(self, step: int) -> None:
        self.current_step = step

    def log(self, module: str, level: str, message: str, **kwargs: Any) -> None:
        entry = AuditEntry(self.current_step, module, level, message, kwargs)
        self.audit_log.append(entry)
        if len(self.audit_log) > self.max_entries:
            self.audit_log.pop(0)

        log_fn = {
            "DEBUG": self._logger.debug,
            "INFO": self._logger.info,
            "WARNING": self._logger.warning,
            "ERROR": self._logger.error,
        }.get(level.upper(), self._logger.info)
        
        meta_str = f" | {kwargs}" if kwargs else ""
        log_fn(f"[{module.upper()}][Step {self.current_step}] {message}{meta_str}")

    def info(self, module: str, message: str, **kwargs: Any) -> None:
        self.log(module, "INFO", message, **kwargs)

    def warning(self, module: str, message: str, **kwargs: Any) -> None:
        self.log(module, "WARNING", message, **kwargs)

    def error(self, module: str, message: str, **kwargs: Any) -> None:
        self.log(module, "ERROR", message, **kwargs)

    def get_recent_entries(self, count: int = 50) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.audit_log[-count:]]
