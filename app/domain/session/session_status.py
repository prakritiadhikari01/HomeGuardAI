from enum import Enum


class SessionStatus(str, Enum):
    """
    Lifecycle of one detected incident.
    """

    CREATED = "CREATED"

    ACTIVE = "ACTIVE"

    WAITING_FOR_ENRICHMENT = "WAITING_FOR_ENRICHMENT"

    COMPLETED = "COMPLETED"

    EXPIRED = "EXPIRED"