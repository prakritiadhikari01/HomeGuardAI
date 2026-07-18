"""
Security mode policy objects — Home.security_mode (NORMAL/AWAY/HIGH/SOS)
drives how aggressively the AI reacts, without scattering `if mode == ...`
checks through AlertProcessor or anywhere else. Each policy answers one
question: given this mode, should this fact become an alert?

house_settings (HomeSettings) still owns the base toggles
(notify_unknown_person, loitering_seconds, etc.) — the policy only
adjusts sensitivity on top of those, it never contradicts a toggle the
person explicitly turned off.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.domain.runtime.setting_runtime import RuntimeHouseSettings


@dataclass(slots=True)
class SecurityPolicy:
    mode: str

    # Multiplies house_settings.loitering_seconds — SOS/AWAY react faster
    # than the base configured threshold; NORMAL uses it as-is.
    loitering_multiplier: float = 1.0

    # Alert immediately on first unknown-person sighting, instead of
    # waiting for house_settings.notify_unknown_person's normal rules.
    alert_on_first_unknown: bool = False

    # Escalate every alert's severity by one step (used by AlertProcessor
    # once it wires severity upgrades through).
    escalate_severity: bool = False

    def loitering_threshold(self, house_settings: RuntimeHouseSettings) -> int:
        if house_settings.loitering_seconds <= 0:
            return 0
        return max(1, int(house_settings.loitering_seconds * self.loitering_multiplier))

    def should_notify_unknown_person(self, house_settings: RuntimeHouseSettings) -> bool:
        if self.alert_on_first_unknown:
            return True
        return house_settings.notify_unknown_person


_POLICIES = {
    "NORMAL": SecurityPolicy(mode="NORMAL", loitering_multiplier=1.0),
    "AWAY": SecurityPolicy(
        mode="AWAY",
        loitering_multiplier=0.5,
        alert_on_first_unknown=True,
    ),
    "HIGH": SecurityPolicy(
        mode="HIGH",
        loitering_multiplier=0.35,
        alert_on_first_unknown=True,
        escalate_severity=True,
    ),
    "SOS": SecurityPolicy(
        mode="SOS",
        loitering_multiplier=0.0,  # 0 -> loitering_threshold() returns 0, alert immediately
        alert_on_first_unknown=True,
        escalate_severity=True,
    ),
}


def get_security_policy(mode: str) -> SecurityPolicy:
    return _POLICIES.get(mode, _POLICIES["NORMAL"])
