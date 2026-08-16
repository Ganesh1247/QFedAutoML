"""
[IMPLEMENTED] Security Audit Logger.
Persists security events, threat detections, auth failures, and privacy budget alerts to security_events table.
"""
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from backend.database.models_orm import SecurityEvent
from backend.monitoring.logger import get_logger

logger = get_logger("SecurityAudit")


class SecurityAuditLogger:
    """Enterprise audit logger for security anomalies and privacy tracking."""

    @staticmethod
    def log_event(
        db: Session,
        event_type: str,
        severity: str = "INFO",
        client_id: str | None = None,
        user_id: int | None = None,
        ip_address: str | None = None,
        details: dict[str, Any] | None = None
    ) -> SecurityEvent:
        """
        Record a security event in the database.
        Severity levels: INFO, WARNING, HIGH, CRITICAL.
        """
        sev_clean = severity.upper().strip()
        event_clean = event_type.upper().strip()

        details_dict = details or {}
        if user_id is not None:
            details_dict["user_id"] = user_id
        if ip_address is not None:
            details_dict["ip_address"] = ip_address

        event = SecurityEvent(
            client_id=client_id,
            event_type=event_clean,
            severity=sev_clean,
            details=details_dict,
            timestamp=datetime.now(UTC)
        )
        db.add(event)
        db.commit()
        db.refresh(event)

        logger.warning(f"SECURITY EVENT [{sev_clean}] {event_clean}: Client={client_id}, Details={details}")
        return event

    @staticmethod
    def get_recent_events(
        db: Session,
        limit: int = 50,
        severity: str | None = None,
        event_type: str | None = None
    ) -> list[SecurityEvent]:
        """Fetch audit log history with optional filters."""
        query = db.query(SecurityEvent)
        if severity:
            query = query.filter(SecurityEvent.severity == severity.upper())
        if event_type:
            query = query.filter(SecurityEvent.event_type == event_type.upper())

        return query.order_by(SecurityEvent.id.desc()).limit(limit).all()


audit_logger = SecurityAuditLogger()
