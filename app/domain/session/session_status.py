from enum import Enum


class SessionStatus(str, Enum):
    """
    Lifecycle of one AI event session.

    One person/object entering the camera creates one session.
    The session lives until the object disappears and all
    enrichment tasks have finished.
    """

    # Session just created from a new ByteTrack track
    CREATED = "CREATED"

    # Object is still visible in camera
    ACTIVE = "ACTIVE"

    # Track ended, waiting for clip generation / VLM summary
    WAITING_FOR_ENRICHMENT = "WAITING_FOR_ENRICHMENT"
    
    # Enrichment tasks started (clip generation / VLM summary)
    ENRICHING = "ENRICHING"

    # Everything finished and PATCH sent to Django
    COMPLETED = "COMPLETED"

    # Session abandoned because of timeout or processing failure
    EXPIRED = "EXPIRED"