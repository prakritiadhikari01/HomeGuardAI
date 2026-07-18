from uuid import uuid4

from app.domain.perception.detection import Detection, ObjectType
from app.domain.perception.track import Track
from app.domain.session.event_session import EventSession
from app.domain.session.session_status import SessionStatus


def create_track():

    detection = Detection(
        object_type=ObjectType.PERSON,
        confidence=0.95,
        bbox=(10, 20, 200, 300),
    )

    return Track(
        track_id=1,
        detection=detection,
    )


def main():

    session = EventSession(
        house_id=uuid4(),
        device_id=uuid4(),
        track=create_track(),
    )

    print("Session created")

    assert session.status == SessionStatus.CREATED

    session.activate()

    assert session.status == SessionStatus.ACTIVE

    print("Activation OK")

    session.wait_for_enrichment()

    assert session.status == SessionStatus.WAITING_FOR_ENRICHMENT

    print("Waiting enrichment OK")

    session.start_enrichment()

    assert session.status == SessionStatus.ENRICHING

    print("Enrichment started")

    session.complete_enrichment()

    assert session.status == SessionStatus.COMPLETED

    print("Completed")

    print("Duration:", session.duration_seconds)

    print("\nALL EVENT SESSION TESTS PASSED")


if __name__ == "__main__":
    main()