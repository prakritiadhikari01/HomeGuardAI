from typing import Optional

from app.domain.alerts.alert_candidate import AlertCandidate
from app.domain.perception.detection import PersonStatus
from app.domain.policies.security_policy import get_security_policy
from app.domain.runtime.setting_runtime import RuntimeHouseSettings
from app.domain.session.event_session import EventSession


class AlertProcessor:
    """Stage 8 (after EnrichmentProcessor) — business rules only. Pure
    domain logic, no networking: returns an AlertCandidate or None;
    PipelineProcessor decides whether to actually call
    DjangoClient.send_alert().

    Consults the house's SecurityPolicy (derived from Home.security_mode)
    on top of the base HomeSettings toggles — AWAY/HIGH/SOS shrink the
    loitering threshold and alert on first sighting; NORMAL just uses
    the settings as configured.

    NOTE: repeated-detection escalation (house_settings.
    repeated_detection_window) needs a per-device history of past
    sessions, which no processor currently tracks — intentionally
    stubbed out rather than silently skipped."""

    def process(
        self,
        session: EventSession,
        house_settings: RuntimeHouseSettings,
        security_mode: str = "NORMAL",
    ) -> Optional[AlertCandidate]:
        if session.alert_generated or session.track is None:
            return None

        policy = get_security_policy(security_mode)
        track = session.track
        alert_type = None

        if track.person_status == PersonStatus.UNKNOWN:
            if policy.should_notify_unknown_person(house_settings):
                alert_type = "UNKNOWN_PERSON"

            loitering_threshold = policy.loitering_threshold(house_settings)
            if loitering_threshold > 0 and session.duration_seconds >= loitering_threshold:
                alert_type = "LOITERING"  # loitering takes priority if both apply

        if alert_type is None:
            return None

        session.alert_generated = True

        return AlertCandidate(
            home_id=session.house_id,
            device_id=session.device_id,
            timeline_event_id=session.timeline_event_id,
            alert_type=alert_type,
            person_label=track.person_label,
            severity="HIGH" if policy.escalate_severity else "NORMAL",
        )
