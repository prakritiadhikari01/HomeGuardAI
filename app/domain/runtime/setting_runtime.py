#runtime_device_settings.py
from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeDeviceSettings:
    """
    Device-specific AI configuration loaded from Django.

    Represents how this camera should behave while the AI engine
    is running.
    """

    enabled: bool

    motion_detection: bool

    human_detection: bool

    face_recognition: bool

    animal_detection: bool

    vehicle_detection: bool

    package_detection: bool

    recording_enabled: bool

    snapshot_enabled: bool

    confidence_threshold: float



@dataclass(slots=True)
class RuntimeHouseSettings:
    """
    House-wide runtime configuration.

    Shared by every camera belonging to the house.
    """

    timeline_enabled: bool

    log_known_member_entry: bool

    log_unknown_person: bool

    notify_unknown_person: bool

    notify_family_members: bool

    emergency_contact_enabled: bool

    escalate_repeated_detection: bool

    ai_summary_enabled: bool

    clip_recording_enabled: bool

    snapshot_enabled: bool

    ignore_animals: bool

    loitering_seconds: int

    repeated_detection_window: int