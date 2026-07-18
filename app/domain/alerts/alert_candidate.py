from dataclasses import dataclass
from typing import Optional
from uuid import UUID


@dataclass(slots=True)
class AlertCandidate:
    """
    Pure data — the domain layer's output when AlertProcessor decides an
    event is worth alerting on. It never sends itself anywhere; only
    PipelineProcessor (composition layer) decides to hand this to
    DjangoClient.

    severity defaults to NORMAL but is left mutable so a later
    EnrichmentProcessor finding (weapon, fire, person jumping a fence)
    can upgrade it before the alert is actually sent — this is why
    alert evaluation happens AFTER enrichment in the pipeline order.
    """

    home_id: UUID
    device_id: UUID
    timeline_event_id: Optional[UUID]
    alert_type: str
    person_label: Optional[str]
    severity: str = "NORMAL"