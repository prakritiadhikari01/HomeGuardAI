from dataclasses import dataclass, field

from app.domain.perception.track import Track


@dataclass(slots=True)
class TrackResult:

    tracks: list[Track] = field(default_factory=list)

    ended_tracks: list[Track] = field(default_factory=list)

    @property
    def active_tracks(self):

        return [
            t
            for t in self.tracks
            if t.active
        ]

    @property
    def person_tracks(self):

        return [
            t
            for t in self.tracks
            if t.is_person
        ]

    @property
    def recognized_tracks(self):

        return [
            t
            for t in self.tracks
            if t.is_recognized
        ]

    @property
    def unknown_tracks(self):

        return [
            t
            for t in self.tracks
            if t.is_unknown_person
        ]