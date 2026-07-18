from uuid import uuid4

from app.domain.perception.detection import Detection, ObjectType
from app.domain.perception.track import Track
from app.domain.perception.track_result import TrackResult
from app.application.event_session_manager import EventSessionManager
from app.domain.session.session_status import SessionStatus


def create_track(track_id):

    detection = Detection(
        object_type=ObjectType.PERSON,
        confidence=0.90,
        bbox=(0, 0, 100, 100),
    )

    return Track(
        track_id=track_id,
        detection=detection,
    )


def main():

    manager = EventSessionManager()

    house_id = uuid4()
    device_id = uuid4()

    print("Creating session...")

    track = create_track(1)

    active, finished = manager.update(
        house_id=house_id,
        device_id=device_id,
        track_result=TrackResult(
            tracks=[track],
            ended_tracks=[],
        ),
    )

    assert len(active) == 1
    assert len(finished) == 0
    assert manager.active_count == 1

    print("Session creation OK")

    print("Updating same session...")

    active, finished = manager.update(
        house_id=house_id,
        device_id=device_id,
        track_result=TrackResult(
            tracks=[track],
            ended_tracks=[],
        ),
    )

    assert manager.active_count == 1

    print("Update OK")

    print("Ending session...")

    active, finished = manager.update(
        house_id=house_id,
        device_id=device_id,
        track_result=TrackResult(
            tracks=[],
            ended_tracks=[track],
        ),
    )

    assert len(active) == 0
    assert len(finished) == 1
    assert finished[0].status == SessionStatus.WAITING_FOR_ENRICHMENT
    assert manager.active_count == 0

    print("Finish OK")

    manager.clear()

    assert manager.active_count == 0

    print("Clear OK")

    print("\nALL EVENT SESSION MANAGER TESTS PASSED")


if __name__ == "__main__":
    main()